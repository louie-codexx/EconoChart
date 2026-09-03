from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any

from econochart.config import load_config, project_path, require
from econochart.data.training import opd_teacher_required_sections
from econochart.evaluation.metrics import aggregate_metrics
from econochart.io import read_jsonl, sha256_file, write_json


def _protocol_issues(row: dict[str, Any]) -> list[str]:
    text = str(row.get("prediction", "")).strip()
    sections = opd_teacher_required_sections(
        str(row.get("task_type", "")),
        view_type=str(row.get("view_type", "")),
    )
    expected_tags = [f"【{section}】" for section in sections]
    issues: list[str] = []
    if not text.startswith(expected_tags[0]):
        issues.append(f"prediction must start with {expected_tags[0]}")
    positions: list[int] = []
    for tag in expected_tags:
        count = text.count(tag)
        if count != 1:
            issues.append(f"{tag} count={count}, expected 1")
        positions.append(text.find(tag))
    if any(position < 0 for position in positions) or positions != sorted(positions):
        issues.append("required section tags are missing or out of order")
    allowed = set(expected_tags)
    unexpected = [tag for tag in ("【结论】", "【数据依据】", "【风险】", "【建议】") if tag not in allowed and tag in text]
    if unexpected:
        issues.append(f"unexpected section tags: {unexpected}")
    markdown_heading = re.compile(
        r"(?m)^\s*(?:#{1,6}\s*|(?:\*\*|__)?[一二三四1234][、.．]\s*)"
        r"(?:结论|数据依据|风险|建议)"
    )
    if markdown_heading.search(text):
        issues.append("Markdown or numbered section heading is forbidden")
    return issues


def _minimum_checks(
    observed: dict[str, Any],
    minimums: dict[str, Any],
    *,
    kind: str,
    scope: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    issues: list[str] = []
    for metric, raw_minimum in minimums.items():
        minimum = float(raw_minimum)
        value = observed.get(metric)
        passed = value is not None and float(value) >= minimum
        checks.append(
            {
                "kind": kind,
                "scope": scope,
                "metric": metric,
                "observed": value,
                "minimum": minimum,
                "passed": passed,
            }
        )
        if not passed:
            issues.append(f"{scope} {metric}={value!r} is below minimum {minimum:.6f}")
    return checks, issues


def evaluate_prompt_smoke(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    predictions_path = project_path(require(config, "inputs.predictions"))
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Teacher prompt smoke predictions do not exist: {predictions_path}")
    rows = list(read_jsonl(predictions_path))
    expected_rows = int(require(config, "inputs.expected_rows"))
    if len(rows) != expected_rows:
        raise ValueError(f"Teacher prompt smoke expected {expected_rows} rows, found {len(rows)}")
    duplicate_count = len(rows) - len({str(row.get("id")) for row in rows})
    if duplicate_count:
        raise ValueError(f"Teacher prompt smoke contains {duplicate_count} duplicate IDs")
    wrong_split = [str(row.get("id", "<missing>")) for row in rows if row.get("split") != "train"]
    if wrong_split:
        raise ValueError(f"Teacher prompt smoke is restricted to split=train; first={wrong_split[0]}")

    task_counts = Counter(str(row.get("task_type")) for row in rows)
    expected_task_counts = {
        str(task): int(count) for task, count in require(config, "inputs.expected_task_counts").items()
    }
    if dict(sorted(task_counts.items())) != dict(sorted(expected_task_counts.items())):
        raise ValueError(
            "Teacher prompt smoke task counts differ from the frozen panel: "
            f"expected={expected_task_counts}, observed={dict(task_counts)}"
        )

    metrics = aggregate_metrics(rows, config.get("reward_weights"))
    checks: list[dict[str, Any]] = []
    issues: list[str] = []
    overall_checks, overall_issues = _minimum_checks(
        metrics["overall"],
        require(config, "gates.minimum_overall"),
        kind="minimum_overall",
        scope="overall",
    )
    checks.extend(overall_checks)
    issues.extend(overall_issues)

    task_slices = metrics["slices"]["task_type"]
    for task_type, minimums in require(config, "gates.minimum_task_metrics").items():
        task_metrics = task_slices.get(str(task_type), {}).get("metrics", {})
        task_checks, task_issues = _minimum_checks(
            task_metrics,
            minimums,
            kind="minimum_task_metric",
            scope=f"task:{task_type}",
        )
        checks.extend(task_checks)
        issues.extend(task_issues)

    for metric, raw_maximum in config.get("gates", {}).get("maximum_efficiency", {}).items():
        maximum = float(raw_maximum)
        value = metrics.get("efficiency", {}).get(metric)
        passed = value is not None and float(value) <= maximum
        checks.append(
            {
                "kind": "maximum_efficiency",
                "scope": "efficiency",
                "metric": metric,
                "observed": value,
                "maximum": maximum,
                "passed": passed,
            }
        )
        if not passed:
            issues.append(f"efficiency {metric}={value!r} exceeds maximum {maximum:.6f}")

    protocol_failures = []
    for row in rows:
        row_issues = _protocol_issues(row)
        if row_issues:
            protocol_failures.append({"id": str(row.get("id")), "issues": row_issues})
    maximum_protocol_failures = int(config.get("gates", {}).get("maximum_protocol_failures", 0))
    protocol_passed = len(protocol_failures) <= maximum_protocol_failures
    checks.append(
        {
            "kind": "protocol_failures",
            "observed": len(protocol_failures),
            "maximum": maximum_protocol_failures,
            "passed": protocol_passed,
        }
    )
    if not protocol_passed:
        issues.append(
            f"strict prompt protocol failed on {len(protocol_failures)} rows "
            f"(maximum {maximum_protocol_failures})"
        )

    passed = not issues
    report = {
        "experiment": config.get("experiment", {}),
        "status": "passed" if passed else "failed",
        "gate": "OPD_TEACHER_PROMPT_SMOKE_PASS" if passed else "OPD_TEACHER_PROMPT_SMOKE_FAIL",
        "input": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "rows": len(rows),
            "split": "train",
            "task_counts": dict(sorted(task_counts.items())),
        },
        "metrics": metrics,
        "gate_evaluation": {
            "passed": passed,
            "checks": checks,
            "issues": issues,
            "protocol_failure_examples": protocol_failures[:10],
        },
        "decision": config.get("decision", {}),
    }
    output = project_path(require(config, "output"))
    if output.exists() and not force:
        raise FileExistsError(f"Prompt-smoke report already exists; use --force only after auditing it: {output}")
    write_json(output, report)
    report["output"] = {"path": str(output), "sha256": sha256_file(output)}
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply frozen absolute gates to a training-only teacher prompt smoke.")
    parser.add_argument("--config", default="configs/opd/teacher_prompt_smoke_v2.yaml")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_prompt_smoke(load_config(args.config), force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
