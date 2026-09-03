from __future__ import annotations

import argparse
import hashlib
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
from econochart.data.training import load_record_sources, summarize_records
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
    def __init__(self, stage: str, config_path: str, *, scope: str) -> None:
        self.payload: dict[str, Any] = {
            "stage": stage,
            "scope": scope,
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


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _records_sha256(records: list[dict[str, Any]]) -> str:
    return _sha256_lines(
        [
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        ]
    )


def _data_checks(report: Report, stage: str, config: dict[str, Any]) -> None:
    if stage == "data":
        output_root = config.get("data", {}).get("output_root")
        report.check("data_output_root", str(project_path(output_root)) if output_root else None)
        return
    seed = int(config.get("seed", 20260821))
    data = config.get("data", {})
    data_summary: dict[str, Any] = {}
    selected_groups: dict[str, list[dict[str, Any]]] = {}
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
            errors: list[str] = []
            schema_error_rows = 0
            for row in rows:
                row_errors = validate_record(row)
                if row_errors:
                    schema_error_rows += 1
                    if len(errors) < 5:
                        errors.extend(f"{row.get('id', '<missing-id>')}: {error}" for error in row_errors)
            if stage == "eval" and group == "test":
                expected_split = str(config.get("evaluation", {}).get("expected_split", "test"))
            else:
                expected_split = "val" if group == "eval" else group
            wrong_split = sum(row.get("split") != expected_split for row in rows)
            source_summary.update(
                {
                    "rows": len(rows),
                    "schema_error_rows": schema_error_rows,
                    "sample_schema_errors": errors[:5],
                    "wrong_split_rows": wrong_split,
                }
            )
            max_samples = source.get("max_samples")
            task_quotas = source.get("task_quotas")
            if task_quotas is not None:
                effective_rows = sum(int(value) for value in task_quotas.values())
            else:
                effective_rows = (
                    len(rows)
                    if max_samples is None or int(max_samples) < 0
                    else min(len(rows), int(max_samples))
                )
            expected_rows = source.get("expected_rows")
            source_summary.update(
                {
                    "effective_rows": effective_rows,
                    "expected_rows": expected_rows,
                    "task_quotas": task_quotas,
                }
            )
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
        group_summary: dict[str, Any] = {"effective_rows_upper_bound": group_rows, "sources": sources}
        expected_split = (
            str(config.get("evaluation", {}).get("expected_split", "test"))
            if stage == "eval" and group == "test"
            else ("val" if group == "eval" else group)
        )
        try:
            selected = load_record_sources(data[group], expected_split=expected_split, seed=seed)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            selected = []
            report.issue(
                "critical",
                "data_selection",
                f"Could not load the effective {group} records: {exc}",
                "Fix the source identity or frozen row counts before launching this run.",
            )
        selected_groups[group] = selected
        if selected:
            ids = [str(row["id"]) for row in selected]
            id_counts = Counter(ids)
            duplicate_ids = sorted(record_id for record_id, count in id_counts.items() if count > 1)
            missing_images = sorted(
                str(row.get("image", ""))
                for row in selected
                if not project_path(str(row.get("image", ""))).is_file()
            )
            dataset_ids: dict[str, list[str]] = {}
            for row in selected:
                dataset_ids.setdefault(str(row.get("dataset", "<missing>")), []).append(str(row["id"]))
            selected_summary = summarize_records(selected)
            selected_summary.update(
                {
                    "unique_ids": len(id_counts),
                    "duplicate_id_count": len(duplicate_ids),
                    "duplicate_id_examples": duplicate_ids[:5],
                    "missing_image_count": len(missing_images),
                    "missing_image_examples": missing_images[:5],
                    "ordered_ids_sha256": _sha256_lines(ids),
                    "ordered_records_sha256": _records_sha256(selected),
                    "dataset_ids_sha256": {
                        name: _sha256_lines(sorted(values)) for name, values in sorted(dataset_ids.items())
                    },
                }
            )
            group_summary["selected"] = selected_summary
            expected_total = data.get("expected_totals", {}).get(group)
            if expected_total is not None and len(selected) != int(expected_total):
                report.issue(
                    "critical",
                    "data",
                    f"Configured {group} total is {int(expected_total)}, but selection produced {len(selected)} rows.",
                    "Restore the frozen source files or correct data.expected_totals before launch.",
                )
            if duplicate_ids:
                report.issue(
                    "critical",
                    "data_identity",
                    f"Effective {group} data contains {len(duplicate_ids)} duplicate record IDs.",
                    "Deduplicate the configured sources before launch.",
                )
            if missing_images:
                report.issue(
                    "critical",
                    "data_images",
                    f"Effective {group} data references {len(missing_images)} missing image files.",
                    "Rebuild or restore every referenced image before launch.",
                )
        data_summary[group] = group_summary
    leakage: dict[str, Any] = {}
    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        if left not in selected_groups or right not in selected_groups:
            continue
        left_ids = {str(row["id"]) for row in selected_groups[left]}
        right_ids = {str(row["id"]) for row in selected_groups[right]}
        left_images = {str(row.get("image", "")) for row in selected_groups[left]}
        right_images = {str(row.get("image", "")) for row in selected_groups[right]}
        id_overlap = sorted(left_ids & right_ids)
        image_overlap = sorted(left_images & right_images)
        name = f"{left}_to_{right}"
        leakage[name] = {
            "id_overlap_count": len(id_overlap),
            "id_overlap_examples": id_overlap[:5],
            "image_path_overlap_count": len(image_overlap),
            "image_path_overlap_examples": image_overlap[:5],
        }
        if id_overlap or image_overlap:
            report.issue(
                "critical",
                "data_leakage",
                f"{left}/{right} overlap detected: ids={len(id_overlap)}, image_paths={len(image_overlap)}.",
                "Restore the frozen split boundary before launch.",
            )
    report.check("data", data_summary)
    report.check("data_leakage", leakage)


def run_preflight(
    config: dict[str, Any],
    *,
    stage: str,
    model_override: str | None = None,
    adapter_override: str | None = None,
    inputs_only: bool = False,
) -> dict[str, Any]:
    scope = "inputs_only" if inputs_only else "launch"
    report = Report(stage, str(config.get("_config_path", "<memory>")), scope=scope)
    # Source-checkout users can invoke preflight before package metadata rejects Python 3.9.
    if sys.version_info < (3, 10):  # noqa: UP036
        report.issue("critical", "python", "Python 3.10+ is required.", "Create a Python 3.10 or 3.11 environment.")
    if not inputs_only:
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
    if inputs_only:
        report.check(
            "launch_readiness",
            {
                "assessed": False,
                "required_next_gate": f"econochart-preflight --stage {stage} without --inputs-only on the GPU host",
            },
        )
    return report.finalize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit AutoDL readiness before data, SFT, GRPO, or evaluation.")
    parser.add_argument("--stage", choices=("data", "sft", "grpo", "eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Override base model path")
    parser.add_argument("--adapter", help="Override adapter path")
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="Audit deterministic data/config identity without claiming package, model, or CUDA launch readiness",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inputs_only and args.stage == "data":
        raise SystemExit("--inputs-only is for sft/grpo/eval configs; stage=data is already data-only")
    report = run_preflight(
        load_config(args.config),
        stage=args.stage,
        model_override=args.model,
        adapter_override=args.adapter,
        inputs_only=args.inputs_only,
    )
    if args.report:
        write_json(project_path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
