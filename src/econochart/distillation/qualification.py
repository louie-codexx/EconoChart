from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from econochart.config import load_config, project_path, require
from econochart.evaluation.compare import compare_rows
from econochart.io import read_json, read_jsonl, sha256_file, write_json


def _load_predictions(
    path_value: str,
    *,
    context: str,
    label: str,
    expected_rows: int,
    expected_split: str,
) -> tuple[Path, list[dict[str, Any]]]:
    path = project_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{context} {label} predictions do not exist: {path}")
    rows = list(read_jsonl(path))
    if len(rows) != expected_rows:
        raise ValueError(f"{context} {label} expected {expected_rows} rows, found {len(rows)}")
    duplicate_count = len(rows) - len({str(row.get("id")) for row in rows})
    if duplicate_count:
        raise ValueError(f"{context} {label} contains {duplicate_count} duplicate IDs")
    wrong_split = [str(row.get("id", "<missing>")) for row in rows if row.get("split") != expected_split]
    if wrong_split:
        raise ValueError(
            f"{context} {label} must contain only split={expected_split}; first={wrong_split[0]}"
        )
    return path, rows


def _metric_summary(comparison: dict[str, Any], metric: str) -> dict[str, Any]:
    summary = comparison.get("overall", {}).get(metric)
    if not isinstance(summary, dict) or not summary.get("rows"):
        raise ValueError(f"Qualification metric {metric!r} is unavailable in paired predictions")
    return summary


def _validate_teacher_candidate_contract(config: dict[str, Any]) -> None:
    inputs = config.get("inputs", {})
    gate_value = inputs.get("prerequisite_gate")
    if gate_value is not None:
        gate_path = project_path(str(gate_value))
        if not gate_path.is_file():
            raise FileNotFoundError(f"Teacher qualification prerequisite gate does not exist: {gate_path}")
        gate_report = read_json(gate_path)
        expected_gate = str(inputs.get("prerequisite_gate_name", "")).strip()
        if not expected_gate:
            raise ValueError("inputs.prerequisite_gate_name is required with prerequisite_gate")
        if gate_report.get("status") != "passed" or gate_report.get("gate") != expected_gate:
            raise ValueError(
                f"Teacher qualification prerequisite gate is not satisfied: expected {expected_gate}, "
                f"observed status={gate_report.get('status')!r}, gate={gate_report.get('gate')!r}"
            )

    manifest_value = inputs.get("candidate_manifest")
    required_profile = inputs.get("required_prompt_profile")
    if manifest_value is None and required_profile is None:
        return
    if manifest_value is None or required_profile is None:
        raise ValueError("candidate_manifest and required_prompt_profile must be configured together")
    manifest_path = project_path(str(manifest_value))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Teacher candidate run manifest does not exist: {manifest_path}")
    manifest = read_json(manifest_path)
    observed_profile = manifest.get("config", {}).get("generation", {}).get("prompt_profile")
    if observed_profile != required_profile:
        raise ValueError(
            "Teacher candidate run manifest has the wrong prompt profile: "
            f"expected {required_profile!r}, observed {observed_profile!r}"
        )


def evaluate_qualification_gates(comparison: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    issues: list[str] = []

    for metric, thresholds in require({"gates": gates}, "gates.required").items():
        summary = _metric_summary(comparison, str(metric))
        delta = float(summary["baseline_to_candidate_delta"])
        probability = float(summary["probability_delta_positive"])
        min_delta = float(thresholds.get("min_delta", float("-inf")))
        min_probability = float(thresholds.get("min_probability_positive", 0.0))
        passed = delta >= min_delta and probability >= min_probability
        checks.append(
            {
                "kind": "required",
                "metric": metric,
                "observed_delta": delta,
                "minimum_delta": min_delta,
                "observed_probability_positive": probability,
                "minimum_probability_positive": min_probability,
                "passed": passed,
            }
        )
        if not passed:
            issues.append(
                f"required metric {metric} failed: delta={delta:.6f} (min {min_delta:.6f}), "
                f"p_positive={probability:.6f} (min {min_probability:.6f})"
            )

    for metric, minimum_delta_value in gates.get("point_estimate_guardrails", {}).items():
        summary = _metric_summary(comparison, str(metric))
        delta = float(summary["baseline_to_candidate_delta"])
        minimum_delta = float(minimum_delta_value)
        passed = delta >= minimum_delta
        checks.append(
            {
                "kind": "point_estimate_guardrail",
                "metric": metric,
                "observed_delta": delta,
                "minimum_delta": minimum_delta,
                "passed": passed,
            }
        )
        if not passed:
            issues.append(
                f"guardrail metric {metric} failed: delta={delta:.6f} (min {minimum_delta:.6f})"
            )

    core_metrics = [str(value) for value in gates.get("core_metrics", [])]
    minimum_positive = int(gates.get("minimum_positive_core_metrics", 0))
    positive_core = [
        metric
        for metric in core_metrics
        if float(_metric_summary(comparison, metric)["baseline_to_candidate_delta"]) > 0
    ]
    core_passed = len(positive_core) >= minimum_positive
    checks.append(
        {
            "kind": "positive_core_count",
            "metrics": core_metrics,
            "positive_metrics": positive_core,
            "observed": len(positive_core),
            "minimum": minimum_positive,
            "passed": core_passed,
        }
    )
    if not core_passed:
        issues.append(
            f"only {len(positive_core)} positive core metrics, below required {minimum_positive}"
        )

    return {"passed": not issues, "checks": checks, "issues": issues}


def run_paired_gate(
    config: dict[str, Any],
    *,
    pass_gate: str,
    fail_gate: str,
    context: str,
    force: bool = False,
) -> dict[str, Any]:
    expected_rows = int(require(config, "inputs.expected_rows"))
    expected_split = str(require(config, "inputs.expected_split"))
    if expected_split != "val":
        raise ValueError(f"{context} is restricted to split=val")
    baseline_path, baseline_rows = _load_predictions(
        require(config, "inputs.baseline"),
        context=context,
        label="baseline",
        expected_rows=expected_rows,
        expected_split=expected_split,
    )
    candidate_path, candidate_rows = _load_predictions(
        require(config, "inputs.candidate"),
        context=context,
        label="candidate",
        expected_rows=expected_rows,
        expected_split=expected_split,
    )
    if baseline_path == candidate_path:
        raise ValueError("Baseline and candidate qualification predictions must be different files")

    comparison = compare_rows(
        baseline_rows,
        candidate_rows,
        seed=int(config.get("seed", 20260821)),
        bootstrap_samples=int(config.get("bootstrap_samples", 5000)),
    )
    gate = evaluate_qualification_gates(comparison, require(config, "gates"))
    report = {
        "experiment": config.get("experiment", {}),
        "status": "passed" if gate["passed"] else "failed",
        "gate": pass_gate if gate["passed"] else fail_gate,
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
                "rows": len(baseline_rows),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
                "rows": len(candidate_rows),
            },
            "split": expected_split,
        },
        "gate_evaluation": gate,
        "comparison": comparison,
        "decision": config.get("decision", {}),
    }
    output = project_path(require(config, "output"))
    if output.exists() and not force:
        raise FileExistsError(f"Paired-gate report already exists; use --force only after auditing it: {output}")
    write_json(output, report)
    report["output"] = {"path": str(output), "sha256": sha256_file(output)}
    return report


def qualify_teacher(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    _validate_teacher_candidate_contract(config)
    return run_paired_gate(
        config,
        pass_gate="OPD_TEACHER_QUALIFICATION_PASS",
        fail_gate="OPD_TEACHER_QUALIFICATION_FAIL",
        context="OPD teacher qualification",
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply the frozen paired gate to an OPD teacher candidate.")
    parser.add_argument("--config", default="configs/opd/teacher_qualification_v1.yaml")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = qualify_teacher(load_config(args.config), force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
