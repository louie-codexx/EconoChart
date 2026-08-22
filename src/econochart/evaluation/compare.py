from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any

from econochart.config import project_path
from econochart.evaluation.metrics import score_row
from econochart.io import read_jsonl, write_json


def _slice_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None and field == "scenario":
        value = row.get("metadata", {}).get("scenario")
    return str(value or "unknown")


def _paired_summary(
    baseline: list[float],
    candidate: list[float],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not baseline:
        return {"rows": 0}
    if len(baseline) != len(candidate):
        raise ValueError("Paired metric vectors must have equal length")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    import numpy as np

    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    deltas = candidate_array - baseline_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)), dtype=np.int32)
    estimates = deltas[indices].mean(axis=1)
    observed = float(deltas.mean())
    return {
        "rows": len(baseline),
        "baseline_mean": round(mean(baseline), 6),
        "candidate_mean": round(mean(candidate), 6),
        "baseline_to_candidate_delta": round(observed, 6),
        "bootstrap_95_ci": [
            round(float(np.quantile(estimates, 0.025)), 6),
            round(float(np.quantile(estimates, 0.975)), 6),
        ],
        "probability_delta_positive": round(float((estimates > 0).mean()), 6),
    }


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int = 20260821,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    baseline = {row["id"]: row for row in baseline_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    if len(baseline) != len(baseline_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("Prediction files must not contain duplicate sample IDs")
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())[:5]
        missing_baseline = sorted(candidate.keys() - baseline.keys())[:5]
        raise ValueError(
            f"Prediction IDs differ; missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    aligned = []
    for record_id in sorted(baseline):
        left, right = baseline[record_id], candidate[record_id]
        for field in ("dataset", "image", "question", "answer", "ground_truth"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Reference field {field!r} differs for {record_id}")
        aligned.append((left, right, score_row(left), score_row(right)))

    metrics = sorted({name for _, _, left, right in aligned for name in left.keys() & right.keys()})
    overall: dict[str, Any] = {}
    for metric in metrics:
        pairs = [
            (float(left_score[metric]), float(right_score[metric]))
            for _, _, left_score, right_score in aligned
            if left_score.get(metric) is not None and right_score.get(metric) is not None
        ]
        overall[metric] = _paired_summary(
            [left for left, _ in pairs],
            [right for _, right in pairs],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )

    slices: dict[str, Any] = {}
    for field in ("dataset", "task_type", "view_type", "industry", "difficulty", "scenario"):
        buckets: dict[str, list[tuple[dict[str, float | None], dict[str, float | None]]]] = defaultdict(list)
        for left, _, left_score, right_score in aligned:
            buckets[_slice_value(left, field)].append((left_score, right_score))
        slices[field] = {}
        for value, pairs in sorted(buckets.items()):
            slice_metrics = {}
            for metric in metrics:
                metric_pairs = [
                    (float(left[metric]), float(right[metric]))
                    for left, right in pairs
                    if left.get(metric) is not None and right.get(metric) is not None
                ]
                if metric_pairs:
                    slice_metrics[metric] = _paired_summary(
                        [left for left, _ in metric_pairs],
                        [right for _, right in metric_pairs],
                        seed=seed,
                        bootstrap_samples=bootstrap_samples,
                    )
            slices[field][value] = {"rows": len(pairs), "metrics": slice_metrics}
    return {
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "paired": True,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "overall": overall,
        "slices": slices,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison for two prediction files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare_rows(
        list(read_jsonl(project_path(args.baseline))),
        list(read_jsonl(project_path(args.candidate))),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_json(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
