from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from econochart.config import ConfigError, load_config, project_path, resolve_adapter_path, resolve_model_path
from econochart.data.schema import validate_record
from econochart.io import read_jsonl, write_json
from econochart.training.common import validate_grpo_batch

REQUIRED_PACKAGES = {
    "sft": {
        "torch": ">=2.6",
        "torchvision": ">=0.21",
        "transformers": ">=4.57.1,<5.16",
        "trl": "==0.28.0",
        "peft": ">=0.18,<0.19",
        "accelerate": ">=1.4,<2",
        "datasets": ">=3,<5",
        "Pillow": ">=10,<13",
        "tensorboard": ">=2.18,<3",
    },
    "grpo": {
        "torch": ">=2.6",
        "torchvision": ">=0.21",
        "transformers": ">=4.57.1,<5.16",
        "trl": "==0.28.0",
        "peft": ">=0.18,<0.19",
        "accelerate": ">=1.4,<2",
        "datasets": ">=3,<5",
        "Pillow": ">=10,<13",
        "tensorboard": ">=2.18,<3",
    },
    "eval": {
        "torch": ">=2.6",
        "torchvision": ">=0.21",
        "transformers": ">=4.57.1,<5.16",
        "peft": ">=0.18,<0.19",
        "datasets": ">=3,<5",
        "Pillow": ">=10,<13",
    },
    "data": {
        "matplotlib": ">=3.8,<4",
        "numpy": ">=1.23.5,<3",
        "Pillow": ">=10,<13",
    },
}


class Report:
    def __init__(self, stage: str, config_path: str) -> None:
        self.payload: dict[str, Any] = {
            "stage": stage,
            "config": config_path,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "checks": {},
            "issues": [],
        }

    def check(self, name: str, value: Any) -> None:
        self.payload["checks"][name] = value

    def issue(self, severity: str, check: str, message: str, action: str) -> None:
        self.payload["issues"].append(
            {"severity": severity, "check": check, "message": message, "action": action}
        )

    def finalize(self) -> dict[str, Any]:
        counts = Counter(issue["severity"] for issue in self.payload["issues"])
        self.payload["severity_counts"] = dict(sorted(counts.items()))
        self.payload["status"] = "failed" if counts.get("critical", 0) else "passed"
        return self.payload


def _package_checks(report: Report, stage: str, config: dict[str, Any]) -> None:
    installed: dict[str, str | None] = {}
    for name, specifier in REQUIRED_PACKAGES[stage].items():
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = None
        installed[name] = value
        if value is None:
            report.issue(
                "critical",
                "package_versions",
                f"Required package {name}{specifier} is not installed.",
                "Install the project extra documented for this stage, then rerun preflight.",
            )
            continue
        try:
            compatible = Version(value) in SpecifierSet(specifier)
        except InvalidVersion:
            compatible = False
        if not compatible:
            report.issue(
                "critical",
                "package_versions",
                f"{name}=={value} does not satisfy the tested range {specifier}.",
                "Use the project lock command; do not mix the legacy AutoDL environment with this run.",
            )

    quantized = str(config.get("model", {}).get("quantization", {}).get("mode", "none")).lower() == "4bit"
    optimizer = str(config.get("training", {}).get("optim", ""))
    if stage in {"sft", "grpo", "eval"} and (quantized or "8bit" in optimizer):
        try:
            installed["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            installed["bitsandbytes"] = None
            report.issue(
                "critical",
                "package_versions",
                "bitsandbytes is required by 4-bit loading or the configured 8-bit optimizer.",
                "Install `bitsandbytes>=0.46,<0.50` on Linux.",
            )
    report.check("packages", installed)


def _cuda_checks(report: Report, stage: str, config: dict[str, Any]) -> None:
    if stage == "data":
        return
    try:
        import torch
    except ImportError:
        return
    cuda = {
        "torch_cuda_version": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        cuda["nvidia_smi"] = [line.strip() for line in nvidia_smi.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        cuda["nvidia_smi"] = None
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        if bool(config.get("training", {}).get("bf16", False)) and not torch.cuda.is_bf16_supported():
            report.issue(
                "critical",
                "cuda",
                "The selected GPU/runtime does not report BF16 support, but bf16=true.",
                "Use a BF16-capable GPU/runtime or create an explicit FP16 experiment config.",
            )
    else:
        report.issue(
            "critical",
            "cuda",
            "CUDA is not available in PyTorch; 4B multimodal training/evaluation is not practical on CPU.",
            "Select the AutoDL GPU instance and install a CUDA-enabled PyTorch build.",
        )
    report.check("cuda", cuda)


def _model_checks(
    report: Report,
    stage: str,
    config: dict[str, Any],
    model_override: str | None,
    adapter_override: str | None,
) -> None:
    if stage == "data":
        return
    try:
        model_path = resolve_model_path(config, model_override)
    except ConfigError as exc:
        report.issue("critical", "model", str(exc), "Set ECONOCHART_MODEL_PATH to the downloaded base model.")
        return
    candidate = Path(model_path).expanduser()
    summary: dict[str, Any] = {"resolved": model_path, "local": candidate.is_dir()}
    if candidate.is_dir():
        required = ["config.json", "preprocessor_config.json", "tokenizer_config.json"]
        missing = [name for name in required if not (candidate / name).is_file()]
        weights = list(candidate.glob("*.safetensors")) + list(candidate.glob("*.bin"))
        summary.update({"missing_metadata": missing, "weight_files": len(weights)})
        if missing or not weights:
            report.issue(
                "critical",
                "model",
                f"Local model directory is incomplete: missing={missing}, weight_files={len(weights)}.",
                "Point ECONOCHART_MODEL_PATH at the complete Qwen3-VL-4B-Instruct directory.",
            )
    else:
        report.issue(
            "medium",
            "model",
            f"{model_path!r} is a Hub ID or unavailable local path; offline readiness cannot be verified.",
            "On AutoDL, set ECONOCHART_MODEL_PATH to the already downloaded local model directory.",
        )
    report.check("model", summary)

    if stage == "grpo" or adapter_override or config.get("model", {}).get("adapter_path"):
        adapter_path = resolve_adapter_path(config, adapter_override)
        if not adapter_path:
            report.issue(
                "critical",
                "adapter",
                "GRPO has no SFT adapter configured.",
                "Run and evaluate SFT first, then set ECONOCHART_ADAPTER_PATH.",
            )
        else:
            adapter = Path(adapter_path).expanduser()
            required = ["adapter_config.json"]
            missing = [name for name in required if not (adapter / name).is_file()]
            weights = list(adapter.glob("adapter_model*.safetensors")) + list(adapter.glob("adapter_model*.bin"))
            report.check(
                "adapter",
                {"resolved": adapter_path, "local": adapter.is_dir(), "missing_metadata": missing, "weights": len(weights)},
            )
            if not adapter.is_dir() or missing or not weights:
                report.issue(
                    "critical",
                    "adapter",
                    f"SFT adapter is missing or incomplete: {adapter_path}.",
                    "Use the final_adapter directory produced by a successful SFT run.",
                )


def _data_checks(report: Report, stage: str, config: dict[str, Any]) -> None:
    if stage == "data":
        output_root = config.get("data", {}).get("output_root")
        report.check("data_output_root", str(project_path(output_root)) if output_root else None)
        return
    data = config.get("data", {})
    data_summary: dict[str, Any] = {}
    for group in ("train", "eval", "test"):
        if group not in data:
            continue
        group_rows = 0
        sources = []
        for source in data[group]:
            path = project_path(source["path"])
            source_summary: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
            if not path.is_file():
                if not source.get("optional", False):
                    report.issue(
                        "critical",
                        "data",
                        f"Required {group} source does not exist: {path}",
                        "Build EconoChart-v2 or prepare the public dataset before this stage.",
                    )
                sources.append(source_summary)
                continue
            rows = list(read_jsonl(path))
            errors = []
            for row in rows[: min(25, len(rows))]:
                errors.extend(validate_record(row))
            if stage == "eval" and group == "test":
                expected_split = str(config.get("evaluation", {}).get("expected_split", "test"))
            else:
                expected_split = "val" if group == "eval" else group
            wrong_split = sum(row.get("split") != expected_split for row in rows)
            source_summary.update({"rows": len(rows), "sample_schema_errors": errors[:5], "wrong_split_rows": wrong_split})
            effective_rows = min(len(rows), int(source.get("max_samples", len(rows))))
            expected_rows = source.get("expected_rows")
            source_summary.update({"effective_rows": effective_rows, "expected_rows": expected_rows})
            group_rows += effective_rows
            if expected_rows is not None and effective_rows != int(expected_rows):
                report.issue(
                    "critical",
                    "data",
                    f"Source {path} expected {int(expected_rows)} effective rows, found {effective_rows}.",
                    "Rebuild the frozen training subsets before launching this run.",
                )
            if errors or wrong_split:
                report.issue(
                    "critical",
                    "data",
                    f"Source {path} failed schema/split checks: sample_errors={errors[:2]}, wrong_split={wrong_split}.",
                    "Rerun dataset validation and do not train on this file.",
                )
            sources.append(source_summary)
        data_summary[group] = {"effective_rows_upper_bound": group_rows, "sources": sources}
    report.check("data", data_summary)


def run_preflight(
    config: dict[str, Any],
    *,
    stage: str,
    model_override: str | None = None,
    adapter_override: str | None = None,
) -> dict[str, Any]:
    report = Report(stage, str(config.get("_config_path", "<memory>")))
    # Source-checkout users can invoke preflight before package metadata rejects Python 3.9.
    if sys.version_info < (3, 10):  # noqa: UP036
        report.issue("critical", "python", "Python 3.10+ is required.", "Create a Python 3.10 or 3.11 environment.")
    _package_checks(report, stage, config)
    _cuda_checks(report, stage, config)
    _model_checks(report, stage, config, model_override, adapter_override)
    _data_checks(report, stage, config)
    if stage == "grpo":
        try:
            report.check("grpo_batch", validate_grpo_batch(config))
        except ConfigError as exc:
            report.issue("critical", "grpo_batch", str(exc), "Fix the batch/generation values before launching GRPO.")
    if stage in {"sft", "grpo"}:
        method = str(config.get("adapter", {}).get("method", "qlora")) if stage == "sft" else "continued_adapter"
        quantized = str(config.get("model", {}).get("quantization", {}).get("mode", "none")) == "4bit"
        if stage == "sft" and method == "full" and quantized:
            report.issue(
                "critical",
                "tuning_method",
                "Full-parameter training cannot update a 4-bit quantized base model.",
                "Use QLoRA, or disable quantization for full tuning on suitable hardware.",
            )
    return report.finalize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit AutoDL readiness before data, SFT, GRPO, or evaluation.")
    parser.add_argument("--stage", choices=("data", "sft", "grpo", "eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Override base model path")
    parser.add_argument("--adapter", help="Override adapter path")
    parser.add_argument("--report", help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_preflight(
        load_config(args.config),
        stage=args.stage,
        model_override=args.model,
        adapter_override=args.adapter,
    )
    if args.report:
        write_json(project_path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
