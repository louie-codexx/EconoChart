from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from econochart.config import load_config, project_path, resolve_adapter_path, resolve_model_path
from econochart.data.training import load_record_sources, summarize_records
from econochart.evaluation.inference import generate_one
from econochart.evaluation.metrics import aggregate_metrics, chartqapro_official_rows
from econochart.io import read_json, read_jsonl, write_json, write_jsonl
from econochart.models.loading import load_adapter, load_base_model, load_processor
from econochart.training.common import resolve_output_dir, save_run_snapshot, seed_everything


def _load_eval_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config.get("seed", 20260821))
    sources = config.get("data", {}).get("test", [])
    expected_split = str(config.get("evaluation", {}).get("expected_split", "test"))
    if expected_split not in {"val", "test"}:
        raise ValueError(f"evaluation.expected_split must be val or test, got {expected_split!r}")
    records = load_record_sources(sources, expected_split=expected_split, seed=seed) if sources else []
    if not records:
        raise ValueError("No evaluation records configured under data.test")
    return records


REFERENCE_FIELDS = ("dataset", "split", "image", "question", "answer", "ground_truth")


def _align_existing_predictions(
    records: list[dict[str, Any]], existing_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected = {row["id"]: row for row in records}
    existing = {row["id"]: row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("Existing prediction file contains duplicate sample IDs")
    unknown = sorted(existing.keys() - expected.keys())
    if unknown:
        raise ValueError(f"Existing prediction file contains unknown sample IDs: {unknown[:5]}")
    for record_id, row in existing.items():
        reference = expected[record_id]
        for field in REFERENCE_FIELDS:
            if row.get(field) != reference.get(field):
                raise ValueError(f"Existing prediction reference field {field!r} differs for {record_id}")
        if "prediction" not in row:
            raise ValueError(f"Existing prediction row has no prediction: {record_id}")
    aligned = [existing[row["id"]] for row in records if row["id"] in existing]
    expected_prefix = [row["id"] for row in records[: len(aligned)]]
    if [row["id"] for row in aligned] != expected_prefix:
        raise ValueError("Existing predictions must be a prefix of the configured deterministic evaluation order")
    return aligned


def _validate_resume_manifest(
    output_dir: Path,
    *,
    config: dict[str, Any],
    model_path: str,
    adapter_path: str | None,
) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cannot safely resume without the original run manifest: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("model") != {"base": model_path, "adapter": adapter_path}:
        raise ValueError("Resume model/adapter differs from the original evaluation run")
    original_config = manifest.get("config", {})
    for key in ("seed", "data", "generation"):
        if original_config.get(key) != config.get(key):
            raise ValueError(f"Resume config field {key!r} differs from the original evaluation run")


def score_predictions(path: str | Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = list(read_jsonl(project_path(path)))
    weights = config.get("evaluation", {}).get("reward_weights")
    metrics = aggregate_metrics(rows, weights)
    write_json(output_dir / "metrics.json", metrics)
    official = chartqapro_official_rows(rows)
    if official:
        write_json(output_dir / "chartqapro_official_predictions.json", official)
    return metrics


def run(
    config: dict[str, Any],
    *,
    model_override: str | None,
    adapter_override: str | None,
    output_override: str | None,
    resume: bool = False,
) -> dict[str, Any]:
    records = _load_eval_records(config)
    model_path = resolve_model_path(config, model_override)
    adapter_path = resolve_adapter_path(config, adapter_override)
    output_dir = resolve_output_dir(config, output_override)
    prediction_path = output_dir / "predictions.jsonl"
    resuming_existing = prediction_path.is_file()
    predictions: list[dict[str, Any]] = []
    if resuming_existing:
        if not resume:
            raise FileExistsError(
                f"Predictions already exist at {prediction_path}. Use --resume only for the same model/config, "
                "or choose a new --output-dir."
            )
        _validate_resume_manifest(
            output_dir,
            config=config,
            model_path=model_path,
            adapter_path=adapter_path,
        )
        predictions = _align_existing_predictions(records, list(read_jsonl(prediction_path)))
        print(f"Resuming evaluation with {len(predictions)}/{len(records)} completed samples")
        if len(predictions) == len(records):
            return score_predictions(prediction_path, output_dir, config)

    seed_everything(int(config.get("seed", 20260821)))
    model, quantized = load_base_model(model_path, config["model"], for_training=False)
    if adapter_path:
        model = load_adapter(model, adapter_path, trainable=False)
    model.eval()
    processor = load_processor(model_path, config["model"], padding_side="left")
    data_summary = summarize_records(records)
    if not resuming_existing:
        save_run_snapshot(
            output_dir,
            stage="evaluation",
            config=config,
            model_path=model_path,
            adapter_path=adapter_path,
            data_summary={**data_summary, "quantized_base": quantized},
        )

    completed_ids = {row["id"] for row in predictions}
    for index, record in enumerate(records, start=1):
        if record["id"] in completed_ids:
            continue
        inference = generate_one(model, processor, record, config.get("generation", {}))
        predictions.append({**record, **inference})
        if len(predictions) % int(config.get("evaluation", {}).get("save_every", 25)) == 0:
            write_jsonl(prediction_path, predictions)
            print(f"Evaluated {index}/{len(records)}")
    write_jsonl(prediction_path, predictions)
    return score_predictions(prediction_path, output_dir, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate base/SFT/GRPO checkpoints on identical fixed records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Override base model path")
    parser.add_argument("--adapter", help="Optional SFT/GRPO adapter path")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--resume", action="store_true", help="Resume a matching interrupted evaluation")
    parser.add_argument("--score-only", help="Score an existing predictions JSONL without loading a model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_dir = resolve_output_dir(config, args.output_dir)
    if args.score_only:
        metrics = score_predictions(args.score_only, output_dir, config)
    else:
        metrics = run(
            config,
            model_override=args.model,
            adapter_override=args.adapter,
            output_override=args.output_dir,
            resume=args.resume,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
