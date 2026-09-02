from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from econochart.config import ROOT, load_config, project_path, require, resolve_model_path
from econochart.data.training import system_prompt_for, user_prompt_for
from econochart.distillation.artifacts import (
    TEACHER_SCORE_SCHEMA_VERSION,
    read_teacher_score_shard,
    record_identity,
    summarize_rollouts,
    validate_resume_snapshot,
    validate_rollout_row,
    write_teacher_score_shard,
)
from econochart.io import read_json, read_jsonl, sha256_file, write_json
from econochart.models.loading import load_base_model, load_processor
from econochart.training.common import save_run_snapshot, seed_everything


def align_rollouts_to_records(
    records: list[dict[str, Any]], rollouts: list[dict[str, Any]], *, round_id: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    record_map = {str(record["id"]): record for record in records}
    if len(record_map) != len(records):
        raise ValueError("OPD teacher prompt source contains duplicate IDs")
    observed_counts: Counter[str] = Counter()
    aligned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rollout in rollouts:
        validate_rollout_row(rollout)
        if int(rollout["round"]) != round_id:
            raise ValueError(f"Teacher score received rollout from round {rollout['round']}, expected {round_id}")
        record_id = str(rollout["record_id"])
        record = record_map.get(record_id)
        if record is None:
            raise ValueError(f"Teacher score rollout references unknown record ID: {record_id}")
        if rollout["record_sha256"] != record_identity(record):
            raise ValueError(f"Teacher score source record identity changed for {record_id}")
        if rollout["image"] != record["image"]:
            raise ValueError(f"Teacher score image path differs from frozen record for {record_id}")
        observed_counts[record_id] += 1
        aligned.append((rollout, record))

    for record_id, record in record_map.items():
        expected = int(record.get("metadata", {}).get("opd", {}).get("rollout_count", 0))
        if observed_counts[record_id] != expected:
            raise ValueError(
                f"Teacher score rollout count mismatch for {record_id}: "
                f"expected {expected}, found {observed_counts[record_id]}"
            )
    return aligned


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


def _load_records(path_value: str, *, expected_rows: int, expected_split: str) -> tuple[Path, list[dict[str, Any]]]:
    path = project_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"OPD teacher source does not exist: {path}")
    records = list(read_jsonl(path))
    if len(records) != expected_rows:
        raise ValueError(f"OPD teacher source expected {expected_rows} rows, found {len(records)}")
    wrong = [str(row.get("id", "<missing>")) for row in records if row.get("split") != expected_split]
    if expected_split != "train" or wrong:
        raise ValueError(f"OPD teacher source must contain only split=train; first={wrong[:1]}")
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


def _score_one(
    model: Any,
    processor: Any,
    rollout: dict[str, Any],
    record: dict[str, Any],
    *,
    top_k: int,
    temperature: float,
    supports_logits_to_keep: bool,
    image_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    import torch

    if temperature <= 0:
        raise ValueError("Teacher scoring temperature must be positive")
    image_path = (ROOT / str(record["image"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Teacher score image missing for {record['id']}: {image_path}")
    if image_path not in image_hash_cache:
        image_hash_cache[image_path] = sha256_file(image_path)
    if image_hash_cache[image_path] != rollout.get("image_sha256"):
        raise ValueError(f"Teacher score image content changed for {record['id']}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor.apply_chat_template(
        _messages(record, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
    if prompt_ids != rollout["prompt_token_ids"]:
        raise ValueError(f"Teacher processor prompt tokens differ from student rollout for {rollout['rollout_key']}")
    completion_ids = [int(value) for value in rollout["completion_token_ids"]]
    device = _device(model)
    completion_tensor = torch.tensor([completion_ids], dtype=inputs["input_ids"].dtype)
    full_input_ids = torch.cat([inputs["input_ids"], completion_tensor], dim=1)
    full_attention_mask = torch.cat(
        [inputs["attention_mask"], torch.ones_like(completion_tensor)], dim=1
    )
    model_inputs = dict(inputs)
    model_inputs["input_ids"] = full_input_ids
    model_inputs["attention_mask"] = full_attention_mask
    model_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in model_inputs.items()}
    forward_kwargs: dict[str, Any] = {**model_inputs, "use_cache": False}
    if supports_logits_to_keep:
        forward_kwargs["logits_to_keep"] = len(completion_ids) + 1
    with torch.inference_mode():
        outputs = model(**forward_kwargs)
    if supports_logits_to_keep:
        logits = outputs.logits[0, :-1, :]
    else:
        start = len(prompt_ids) - 1
        logits = outputs.logits[0, start : start + len(completion_ids), :]
    if logits.shape[0] != len(completion_ids):
        raise ValueError(
            f"Teacher logit alignment failed for {rollout['rollout_key']}: "
            f"expected {len(completion_ids)}, found {logits.shape[0]}"
        )
    if top_k < 1 or top_k > logits.shape[-1]:
        raise ValueError(f"Teacher top_k must be within [1, {logits.shape[-1]}], got {top_k}")
    logprobs = torch.log_softmax(logits.float() / temperature, dim=-1)
    top_values, top_ids = logprobs.topk(top_k, dim=-1)
    return {
        "schema_version": TEACHER_SCORE_SCHEMA_VERSION,
        "rollout_key": rollout["rollout_key"],
        "prompt_token_ids_sha256": rollout["prompt_token_ids_sha256"],
        "completion_token_ids_sha256": rollout["completion_token_ids_sha256"],
        "completion_token_ids": completion_ids,
        "topk_token_ids": top_ids.cpu().to(torch.int64).tolist(),
        "topk_logprobs": top_values.cpu().to(torch.float32).tolist(),
    }


def _restored_shard_summary(path: Path, expected_keys: list[str]) -> dict[str, Any]:
    rows = read_teacher_score_shard(path)
    actual_keys = [str(row["rollout_key"]) for row in rows]
    if actual_keys != expected_keys:
        raise ValueError(f"Existing teacher score shard has unexpected rollout keys: {path}")
    import numpy as np

    mass_vectors = [np.exp(row["topk_logprobs"]).sum(axis=1) for row in rows]
    masses = np.concatenate(mass_vectors)
    return {
        "path": str(path),
        "rows": len(rows),
        "tokens": sum(len(row["completion_token_ids"]) for row in rows),
        "top_k": len(rows[0]["topk_token_ids"][0]),
        "topk_mass_mean": round(float(masses.mean()), 6),
        "topk_mass_min": round(float(masses.min()), 6),
        "sha256": sha256_file(path),
    }


def run(
    config: dict[str, Any],
    *,
    model_override: str | None,
    output_override: str | None,
    resume: bool,
) -> dict[str, Any]:
    seed = int(config.get("seed", 20260821))
    round_id = int(require(config, "experiment.round"))
    prompt_source = require(config, "data.prompts")
    prompt_path, records = _load_records(
        require(prompt_source, "path"),
        expected_rows=int(require(prompt_source, "expected_rows")),
        expected_split=str(require(prompt_source, "expected_split")),
    )
    rollout_source = require(config, "data.rollouts")
    rollout_path = project_path(require(rollout_source, "path"))
    rollout_manifest_path = project_path(require(rollout_source, "manifest"))
    if not rollout_path.is_file() or not rollout_manifest_path.is_file():
        raise FileNotFoundError(f"Student rollout artifacts are missing: {rollout_path}, {rollout_manifest_path}")
    rollouts = list(read_jsonl(rollout_path))
    expected_rollouts = int(require(rollout_source, "expected_rows"))
    if len(rollouts) != expected_rollouts:
        raise ValueError(f"Teacher score expected {expected_rollouts} rollouts, found {len(rollouts)}")
    summarize_rollouts(rollouts)
    aligned = align_rollouts_to_records(records, rollouts, round_id=round_id)

    prerequisites = {
        label: _require_gate(entry, label=label)
        for label, entry in require(config, "prerequisites").items()
    }
    scoring = require(config, "scoring")
    if int(require(scoring, "expected_rollouts")) != len(aligned):
        raise ValueError("scoring.expected_rollouts differs from the aligned rollout count")
    output_dir = project_path(output_override or require(scoring, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / str(require(scoring, "manifest_file"))
    snapshot_path = output_dir / "run_manifest.json"
    model_path = resolve_model_path(config, model_override)
    resume_data = {
        "prompts": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "rows": len(records),
        },
        "rollouts": {
            "path": str(rollout_path),
            "sha256": sha256_file(rollout_path),
            "rows": len(rollouts),
        },
        "rollout_manifest_sha256": sha256_file(rollout_manifest_path),
        "prerequisites": prerequisites,
    }
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Teacher score manifest already exists: {manifest_path}")
        validate_resume_snapshot(
            output_dir,
            stage="opd_teacher_score",
            config=config,
            model_path=model_path,
            adapter_path=None,
            data_summary=resume_data,
        )
        existing_manifest = read_json(manifest_path)
        if existing_manifest.get("rollouts", {}).get("sha256") != sha256_file(rollout_path):
            raise ValueError("Existing teacher score manifest belongs to different student rollouts")
        if existing_manifest.get("model", {}).get("teacher") != model_path:
            raise ValueError("Existing teacher score manifest belongs to a different teacher")
        if (
            existing_manifest.get("gate") != "OPD_TEACHER_SCORE_ARTIFACT_PASS"
            or existing_manifest.get("status") != "passed"
        ):
            raise ValueError("Existing teacher score manifest is missing its PASS gate")
        for shard in existing_manifest.get("shards", []):
            path = Path(str(shard["path"])).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != shard.get("sha256"):
                raise ValueError(f"Existing teacher score shard is missing or changed: {path}")
        existing_manifest["manifest_sha256"] = sha256_file(manifest_path)
        return existing_manifest
    existing_shards = sorted(output_dir.glob("teacher_scores_*.npz"))
    if existing_shards and not resume:
        raise FileExistsError(f"Teacher score shards already exist; use --resume: {existing_shards[0]}")

    if existing_shards or resume:
        validate_resume_snapshot(
            output_dir,
            stage="opd_teacher_score",
            config=config,
            model_path=model_path,
            adapter_path=None,
            data_summary=resume_data,
        )
    else:
        if snapshot_path.exists():
            raise FileExistsError(f"OPD teacher score run snapshot already exists; use --resume: {snapshot_path}")
        save_run_snapshot(
            output_dir,
            stage="opd_teacher_score",
            config=config,
            model_path=model_path,
            adapter_path=None,
            data_summary=resume_data,
        )
    seed_everything(seed)
    model, quantized = load_base_model(model_path, config["model"], for_training=False)
    if quantized:
        raise ValueError("Formal OPD teacher scoring requires the unquantized teacher configured by the audit")
    model.eval()
    processor = load_processor(model_path, config["model"], padding_side="left")
    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    top_k = int(require(scoring, "top_k"))
    temperature = float(require(scoring, "temperature"))
    shard_size = int(require(scoring, "shard_size"))
    if shard_size < 1:
        raise ValueError("Teacher score shard_size must be positive")

    shard_summaries: list[dict[str, Any]] = []
    image_hash_cache: dict[Path, str] = {}
    for start in range(0, len(aligned), shard_size):
        stop = min(start + shard_size, len(aligned))
        shard_path = output_dir / f"teacher_scores_{start:05d}_{stop - 1:05d}.npz"
        expected_keys = [str(rollout["rollout_key"]) for rollout, _ in aligned[start:stop]]
        if shard_path.exists():
            if not resume:
                raise FileExistsError(f"Teacher score shard already exists: {shard_path}")
            shard_summaries.append(_restored_shard_summary(shard_path, expected_keys))
            continue
        score_rows = [
            _score_one(
                model,
                processor,
                rollout,
                record,
                top_k=top_k,
                temperature=temperature,
                supports_logits_to_keep=supports_logits_to_keep,
                image_hash_cache=image_hash_cache,
            )
            for rollout, record in aligned[start:stop]
        ]
        summary = write_teacher_score_shard(shard_path, score_rows)
        summary["sha256"] = sha256_file(shard_path)
        shard_summaries.append(summary)
        print(f"Teacher-scored OPD rollouts {stop}/{len(aligned)}")

    total_tokens = sum(int(summary["tokens"]) for summary in shard_summaries)
    mean_mass = (
        sum(float(summary["topk_mass_mean"]) * int(summary["tokens"]) for summary in shard_summaries)
        / max(1, total_tokens)
    )
    minimum_mass = float(require(scoring, "minimum_mean_topk_mass"))
    if mean_mass < minimum_mass:
        raise ValueError(
            f"Teacher top-k mean mass {mean_mass:.6f} is below frozen minimum {minimum_mass:.6f}; "
            "do not train on this artifact"
        )
    manifest = {
        "schema_version": TEACHER_SCORE_SCHEMA_VERSION,
        "experiment": config.get("experiment", {}),
        "model": {"teacher": model_path, "quantized": quantized},
        "prompts": {"path": str(prompt_path), "sha256": sha256_file(prompt_path), "rows": len(records)},
        "rollouts": {
            "path": str(rollout_path),
            "sha256": sha256_file(rollout_path),
            "manifest_sha256": sha256_file(rollout_manifest_path),
            "rows": len(rollouts),
        },
        "prerequisites": prerequisites,
        "scoring": {
            "temperature": temperature,
            "top_k": top_k,
            "supports_logits_to_keep": supports_logits_to_keep,
            "mean_topk_mass": round(mean_mass, 6),
            "minimum_mean_topk_mass": minimum_mass,
            "tokens": total_tokens,
        },
        "shards": shard_summaries,
        "gate": "OPD_TEACHER_SCORE_ARTIFACT_PASS",
        "status": "passed",
    }
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score image-aware student rollouts with the frozen OPD teacher.")
    parser.add_argument("--config", default="configs/opd/round1_teacher_score_v1.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(
        load_config(args.config),
        model_override=args.model,
        output_override=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
