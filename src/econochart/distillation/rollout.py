from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from econochart.config import ROOT, load_config, project_path, require, resolve_adapter_path, resolve_model_path
from econochart.data.training import system_prompt_for, user_prompt_for
from econochart.distillation.artifacts import (
    ROLLOUT_SCHEMA_VERSION,
    record_identity,
    sha256_json,
    summarize_rollouts,
    validate_adapter_snapshot,
    validate_resume_snapshot,
    validate_rollout_row,
)
from econochart.io import read_json, read_jsonl, sha256_file, stable_seed, write_json, write_jsonl
from econochart.models.loading import load_adapter, load_base_model, load_processor
from econochart.training.common import save_run_snapshot, seed_everything


def expand_rollout_jobs(records: list[dict[str, Any]], *, round_id: int, seed: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {}).get("opd", {})
        if int(metadata.get("round", 0)) != round_id:
            raise ValueError(f"OPD record {record.get('id')} does not belong to round {round_id}")
        rollout_count = int(metadata.get("rollout_count", 0))
        if rollout_count not in {1, 2}:
            raise ValueError(f"OPD record {record.get('id')} must request one or two rollouts")
        for rollout_index in range(rollout_count):
            key = f"{record['id']}:round{round_id}:sample{rollout_index}"
            jobs.append(
                {
                    "rollout_key": key,
                    "rollout_index": rollout_index,
                    "rollout_seed": stable_seed(f"opd:rollout:{key}", seed) % (2**31),
                    "record": record,
                }
            )
    return jobs


def validate_image_token_presence(prompt_token_ids: list[int], image_token_id: int | None) -> int:
    if image_token_id is None or image_token_id < 0:
        raise ValueError("Qwen3-VL image token ID is unavailable")
    count = prompt_token_ids.count(image_token_id)
    if count < 1:
        raise ValueError("Image-aware OPD prompt contains no image tokens")
    return count


def _load_records(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    source = require(config, "data.prompts")
    path = project_path(require(source, "path"))
    if not path.is_file():
        raise FileNotFoundError(f"OPD rollout prompt source does not exist: {path}")
    records = list(read_jsonl(path))
    expected_rows = int(require(source, "expected_rows"))
    if len(records) != expected_rows:
        raise ValueError(f"OPD rollout source expected {expected_rows} rows, found {len(records)}")
    expected_split = str(require(source, "expected_split"))
    wrong_split = [str(row.get("id", "<missing>")) for row in records if row.get("split") != expected_split]
    if expected_split != "train" or wrong_split:
        raise ValueError(f"OPD rollout source must contain only split=train; first={wrong_split[:1]}")
    if len(records) != len({str(row.get("id")) for row in records}):
        raise ValueError("OPD rollout prompt source contains duplicate IDs")
    return path, records


def _require_gate(entry: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = project_path(require(entry, "path"))
    if not path.is_file():
        raise FileNotFoundError(f"Missing OPD {label} prerequisite report: {path}")
    report = read_json(path)
    expected_gate = str(require(entry, "expected_gate"))
    if report.get("gate") != expected_gate or report.get("status") != "passed":
        raise ValueError(
            f"OPD {label} prerequisite failed: expected {expected_gate}/passed, "
            f"found {report.get('gate')}/{report.get('status')}"
        )
    return {"path": str(path), "sha256": sha256_file(path), "gate": expected_gate}


def _messages(record: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt_for(record)}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt_for(record)},
            ],
        },
    ]


def _device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _image_token_id(model: Any, processor: Any) -> int | None:
    value = getattr(model.config, "image_token_id", None)
    if value is not None:
        return int(value)
    token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return int(token_id) if isinstance(token_id, int) and token_id >= 0 else None


def _generate_job(
    model: Any,
    processor: Any,
    job: dict[str, Any],
    generation: dict[str, Any],
    *,
    round_id: int,
    image_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    import torch

    record = job["record"]
    image_path = (ROOT / str(record["image"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"OPD image missing for {record['id']}: {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor.apply_chat_template(
        _messages(record, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_token_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
    image_token_count = validate_image_token_presence(prompt_token_ids, _image_token_id(model, processor))
    inputs = inputs.to(_device(model))
    rollout_seed = int(job["rollout_seed"])
    torch.manual_seed(rollout_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rollout_seed)
        torch.cuda.synchronize()

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation.get("max_new_tokens", 256)),
        "do_sample": bool(generation.get("do_sample", True)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(generation.get("temperature", 0.8))
        kwargs["top_p"] = float(generation.get("top_p", 0.95))
        configured_top_k = int(generation.get("top_k", 0))
        if configured_top_k > 0:
            kwargs["top_k"] = configured_top_k
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    completion = [int(value) for value in generated[0, len(prompt_token_ids) :].tolist()]
    if not completion:
        raise ValueError(f"OPD student produced no completion tokens for {job['rollout_key']}")
    prediction = processor.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    if image_path not in image_hash_cache:
        image_hash_cache[image_path] = sha256_file(image_path)
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    row = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollout_key": str(job["rollout_key"]),
        "record_id": str(record["id"]),
        "record_sha256": record_identity(record),
        "dataset": str(record["dataset"]),
        "split": str(record["split"]),
        "image": str(record["image"]),
        "image_sha256": image_hash_cache[image_path],
        "round": round_id,
        "rollout_index": int(job["rollout_index"]),
        "rollout_seed": rollout_seed,
        "prediction": prediction,
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": sha256_json(prompt_token_ids),
        "completion_token_ids": completion,
        "completion_token_ids_sha256": sha256_json(completion),
        "image_token_count": image_token_count,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": len(completion),
        "truncated": len(completion) >= kwargs["max_new_tokens"] and not (eos_ids & set(completion)),
        "latency_seconds": round(latency, 6),
    }
    validate_rollout_row(row)
    return row


def _validate_resume(existing: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> None:
    if len(existing) > len(jobs):
        raise ValueError("Existing OPD rollout file is longer than the configured job list")
    for row, job in zip(existing, jobs, strict=False):
        validate_rollout_row(row)
        record = job["record"]
        expected = {
            "rollout_key": str(job["rollout_key"]),
            "record_id": str(record["id"]),
            "record_sha256": record_identity(record),
            "image": str(record["image"]),
            "round": int(record.get("metadata", {}).get("opd", {}).get("round", 0)),
            "rollout_index": int(job["rollout_index"]),
            "rollout_seed": int(job["rollout_seed"]),
        }
        changed = [field for field, value in expected.items() if row.get(field) != value]
        if changed:
            raise ValueError(
                f"Existing OPD rollout identity differs for {job['rollout_key']}: {changed}"
            )
    expected_keys = [str(job["rollout_key"]) for job in jobs[: len(existing)]]
    actual_keys = [str(row["rollout_key"]) for row in existing]
    if actual_keys != expected_keys:
        raise ValueError("Existing OPD rollouts must be an exact prefix of the deterministic job order")


def _write_rollout_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def run(
    config: dict[str, Any],
    *,
    model_override: str | None,
    adapter_override: str | None,
    output_override: str | None,
    resume: bool,
) -> dict[str, Any]:
    seed = int(config.get("seed", 20260821))
    round_id = int(require(config, "experiment.round"))
    source_path, records = _load_records(config)
    jobs = expand_rollout_jobs(records, round_id=round_id, seed=seed)
    expected_rollouts = int(require(config, "rollout.expected_rollouts"))
    if len(jobs) != expected_rollouts:
        raise ValueError(f"OPD round {round_id} expected {expected_rollouts} rollout jobs, found {len(jobs)}")

    prerequisites = {
        label: _require_gate(entry, label=label)
        for label, entry in require(config, "prerequisites").items()
    }
    opd_manifest_path = project_path(require(config, "data.opd_manifest"))
    if not opd_manifest_path.is_file():
        raise FileNotFoundError(f"OPD data manifest does not exist: {opd_manifest_path}")

    output_dir = project_path(output_override or require(config, "rollout.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / str(require(config, "rollout.predictions_file"))
    manifest_path = output_dir / str(require(config, "rollout.manifest_file"))
    snapshot_path = output_dir / "run_manifest.json"
    model_path = resolve_model_path(config, model_override)
    adapter_path = resolve_adapter_path(config, adapter_override)
    if not adapter_path:
        raise ValueError("OPD rollout must start from the frozen GRPO student adapter")
    adapter_identity = validate_adapter_snapshot(
        adapter_path,
        str(require(config, "model.expected_adapter_sha256")),
    )
    resume_data = {
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "records": len(records),
        "rollouts": len(jobs),
        "opd_manifest_sha256": sha256_file(opd_manifest_path),
        "student_adapter": adapter_identity,
        "prerequisites": prerequisites,
    }
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"OPD rollout manifest already exists: {manifest_path}")
        validate_resume_snapshot(
            output_dir,
            stage="opd_student_rollout",
            config=config,
            model_path=model_path,
            adapter_path=adapter_path,
            data_summary=resume_data,
        )
        manifest = read_json(manifest_path)
        if not rollout_path.is_file() or manifest.get("rollouts", {}).get("sha256") != sha256_file(
            rollout_path
        ):
            raise ValueError("Completed OPD rollout manifest does not match its rollout file")
        if (
            manifest.get("gate") != "OPD_STUDENT_ROLLOUT_ARTIFACT_PASS"
            or manifest.get("status") != "passed"
        ):
            raise ValueError("Completed OPD rollout manifest is missing its PASS gate")
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        return manifest
    existing: list[dict[str, Any]] = []
    if rollout_path.exists():
        if not resume:
            raise FileExistsError(f"OPD rollouts already exist at {rollout_path}; use --resume for the same run")
        existing = list(read_jsonl(rollout_path))
        _validate_resume(existing, jobs)
        validate_resume_snapshot(
            output_dir,
            stage="opd_student_rollout",
            config=config,
            model_path=model_path,
            adapter_path=adapter_path,
            data_summary=resume_data,
        )
    elif resume:
        validate_resume_snapshot(
            output_dir,
            stage="opd_student_rollout",
            config=config,
            model_path=model_path,
            adapter_path=adapter_path,
            data_summary=resume_data,
        )
    elif snapshot_path.exists():
        raise FileExistsError(f"OPD rollout run snapshot already exists; use --resume: {snapshot_path}")
    if not existing and not resume:
        save_run_snapshot(
            output_dir,
            stage="opd_student_rollout",
            config=config,
            model_path=model_path,
            adapter_path=adapter_path,
            data_summary=resume_data,
        )

    seed_everything(seed)
    model, quantized = load_base_model(model_path, config["model"], for_training=False)
    model = load_adapter(model, adapter_path, trainable=False)
    model.eval()
    processor = load_processor(model_path, config["model"], padding_side="left")
    rows = list(existing)
    image_hash_cache: dict[Path, str] = {}
    save_every = int(config.get("rollout", {}).get("save_every", 10))
    for index, job in enumerate(jobs[len(rows) :], start=len(rows) + 1):
        rows.append(
            _generate_job(
                model,
                processor,
                job,
                config.get("generation", {}),
                round_id=round_id,
                image_hash_cache=image_hash_cache,
            )
        )
        if len(rows) % save_every == 0:
            _write_rollout_checkpoint(rollout_path, rows)
            print(f"Generated OPD rollouts {index}/{len(jobs)}")
    _write_rollout_checkpoint(rollout_path, rows)
    summary = summarize_rollouts(rows)
    manifest = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "experiment": config.get("experiment", {}),
        "model": {
            "base": model_path,
            "adapter": adapter_identity,
            "quantized_base": quantized,
        },
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "opd_manifest": {"path": str(opd_manifest_path), "sha256": sha256_file(opd_manifest_path)},
        "prerequisites": prerequisites,
        "generation": config.get("generation", {}),
        "summary": summary,
        "rollouts": {"path": str(rollout_path), "sha256": sha256_file(rollout_path)},
        "gate": "OPD_STUDENT_ROLLOUT_ARTIFACT_PASS",
        "status": "passed",
    }
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate auditable image-aware on-policy student rollouts.")
    parser.add_argument("--config", default="configs/opd/round1_rollout_v1.yaml")
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(
        load_config(args.config),
        model_override=args.model,
        adapter_override=args.adapter,
        output_override=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
