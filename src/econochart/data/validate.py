from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from econochart.config import ROOT, project_path
from econochart.data.schema import parse_ground_truth, validate_record
from econochart.io import read_json, read_jsonl, sha256_file, write_json

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    check: str
    evidence: str
    remediation: str


def _extract_numbers(text: str) -> list[float]:
    return [float(match.group()) for match in NUMBER_PATTERN.finditer(text.replace(",", ""))]


def _target_in_answer(target: dict[str, Any], answer: str) -> bool:
    value = float(target["value"])
    tolerance = float(target.get("tolerance", max(0.2, abs(value) * 0.005)))
    return any(math.isclose(candidate, value, abs_tol=tolerance, rel_tol=0.0) for candidate in _extract_numbers(answer))


def _add(
    issues: list[ValidationIssue], severity: str, check: str, evidence: str, remediation: str
) -> None:
    issues.append(ValidationIssue(severity, check, evidence, remediation))


def _read_splits(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    annotations = dataset_root / "annotations"
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = annotations / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation split: {path}")
        splits[split] = list(read_jsonl(path))
    return splits


def _check_images(
    rows: list[dict[str, Any]],
    issues: list[ValidationIssue],
    *,
    full_scan: bool,
    expected_size: tuple[int, int] | None,
) -> dict[str, Any]:
    unique_paths = sorted({row["image"] for row in rows})
    missing = [value for value in unique_paths if not (ROOT / value).is_file()]
    if missing:
        _add(
            issues,
            "critical",
            "image_referential_integrity",
            f"{len(missing)}/{len(unique_paths)} referenced images are missing; first={missing[0]}",
            "Rebuild images and annotations in one command; do not move generated images independently.",
        )

    existing = [ROOT / value for value in unique_paths if (ROOT / value).is_file()]
    if not full_scan and len(existing) > 100:
        stride = max(1, len(existing) // 100)
        existing = existing[::stride][:100]

    dimensions: Counter[str] = Counter()
    open_errors: list[str] = []
    try:
        from PIL import Image

        for path in existing:
            try:
                with Image.open(path) as image:
                    image.load()
                    dimensions[f"{image.width}x{image.height}"] += 1
                    if image.mode not in {"RGB", "RGBA"}:
                        open_errors.append(f"{path.name}: unexpected mode {image.mode}")
            except Exception as exc:  # noqa: BLE001 - validator should report every corrupt image
                open_errors.append(f"{path.name}: {exc}")
    except ImportError:
        _add(
            issues,
            "medium",
            "image_decode",
            "Pillow is not installed; image decode checks were skipped.",
            "Install the data extra with `pip install -e '.[data]'` and rerun validation.",
        )

    if open_errors:
        _add(
            issues,
            "critical",
            "image_decode",
            f"{len(open_errors)} sampled images failed validation; first={open_errors[0]}",
            "Regenerate the affected charts and run a full image scan.",
        )
    if expected_size and dimensions:
        expected_key = f"{expected_size[0]}x{expected_size[1]}"
        wrong = {key: value for key, value in dimensions.items() if key != expected_key}
        if wrong:
            _add(
                issues,
                "high",
                "image_shape",
                f"Expected {expected_size[0]}x{expected_size[1]}, observed incompatible dimensions: {wrong}",
                "Use fixed figsize/dpi and avoid tight bounding-box cropping.",
            )
    return {
        "referenced_images": len(unique_paths),
        "missing_images": len(missing),
        "decoded_images": len(existing),
        "dimensions": dict(dimensions),
        "full_scan": full_scan,
    }


def validate_dataset(dataset_root: str | Path, *, full_image_scan: bool = False) -> dict[str, Any]:
    root = project_path(dataset_root)
    issues: list[ValidationIssue] = []
    manifest_path = root / "manifest.json"
    companies_path = root / "raw" / "companies.jsonl"
    if not manifest_path.is_file() or not companies_path.is_file():
        raise FileNotFoundError(f"Expected manifest.json and raw/companies.jsonl below {root}")

    manifest = read_json(manifest_path)
    companies = list(read_jsonl(companies_path))
    rows_by_split = _read_splits(root)
    all_rows = [row for rows in rows_by_split.values() for row in rows]

    schema_errors: list[str] = []
    for row in all_rows:
        errors = validate_record(row)
        if errors:
            schema_errors.append(f"{row.get('id', '<missing-id>')}: {'; '.join(errors)}")
    if schema_errors:
        _add(
            issues,
            "critical",
            "schema",
            f"{len(schema_errors)} records violate the schema; first={schema_errors[0]}",
            "Fix the generator or public adapter before starting any training.",
        )

    ids = [row["id"] for row in all_rows]
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        _add(
            issues,
            "critical",
            "record_uniqueness",
            f"Found {duplicate_ids} duplicate record IDs.",
            "Build IDs from dataset, entity, chart, and task keys and enforce uniqueness during generation.",
        )

    entity_sets = {split: {row["entity_id"] for row in rows} for split, rows in rows_by_split.items()}
    chart_sets = {split: {row["chart_id"] for row in rows} for split, rows in rows_by_split.items()}
    image_sets = {split: {row["image"] for row in rows} for split, rows in rows_by_split.items()}
    leakage: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for grain, sets in (("entity", entity_sets), ("chart", chart_sets), ("image", image_sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                leakage[f"{grain}:{left}-{right}"] = sorted(overlap)[:10]
    if leakage:
        _add(
            issues,
            "critical",
            "split_leakage",
            json.dumps(leakage, ensure_ascii=False),
            "Split by entity before rendering charts or deriving questions; rebuild all downstream files.",
        )

    rows_per_chart = Counter(row["chart_id"] for row in all_rows)
    unexpected_chart_grain = {key: value for key, value in rows_per_chart.items() if value != 4}
    if unexpected_chart_grain:
        first = next(iter(unexpected_chart_grain.items()))
        _add(
            issues,
            "high",
            "chart_question_grain",
            f"{len(unexpected_chart_grain)} charts do not have exactly 4 aligned tasks; first={first}",
            "Generate the complete view-specific task set atomically for every chart.",
        )

    answers_by_chart: dict[str, set[str]] = defaultdict(set)
    questions_by_chart: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        answers_by_chart[row["chart_id"]].add(row["answer"])
        questions_by_chart[row["chart_id"]].add(row["question"])
    collapsed_answers = [chart for chart, answers in answers_by_chart.items() if len(answers) < rows_per_chart[chart]]
    collapsed_questions = [chart for chart, questions in questions_by_chart.items() if len(questions) < rows_per_chart[chart]]
    if collapsed_answers:
        _add(
            issues,
            "high",
            "answer_task_alignment",
            f"{len(collapsed_answers)} charts reuse an identical answer across distinct tasks.",
            "Use task-specific answer builders and include the facts requested by each question.",
        )
    if collapsed_questions:
        _add(
            issues,
            "high",
            "question_uniqueness",
            f"{len(collapsed_questions)} charts contain duplicate questions.",
            "Ensure every chart-task key maps to a distinct question.",
        )

    unique_answer_ratio = len({row["answer"] for row in all_rows}) / max(1, len(all_rows))
    if unique_answer_ratio < 0.80:
        _add(
            issues,
            "high",
            "answer_diversity",
            f"Only {unique_answer_ratio:.2%} of answers are unique.",
            "Increase grounded numeric and task-specific variation; inspect templates for collapse.",
        )

    missing_target_values: list[str] = []
    for row in all_rows:
        ground_truth = parse_ground_truth(row["ground_truth"])
        for target in ground_truth.get("numeric_targets", []):
            if not _target_in_answer(target, row["answer"]):
                missing_target_values.append(f"{row['id']}:{target['name']}={target['value']}")
    if missing_target_values:
        _add(
            issues,
            "critical",
            "reference_ground_truth_consistency",
            f"{len(missing_target_values)} numeric targets are absent from their reference answer; first={missing_target_values[0]}",
            "Generate the answer and ground truth from the same fact object and add a regression test.",
        )

    company_ids = [company["entity_id"] for company in companies]
    if len(company_ids) != len(set(company_ids)):
        _add(
            issues,
            "critical",
            "company_uniqueness",
            f"Found {len(company_ids) - len(set(company_ids))} duplicate company IDs.",
            "Fix company ID construction and rebuild splits.",
        )
    profit_identity_errors = 0
    for company in companies:
        metrics = company["metrics"]
        for revenue, cost, profit in zip(
            metrics["revenue_million"], metrics["cost_million"], metrics["profit_million"]
        ):
            if not math.isclose(revenue - cost, profit, abs_tol=0.11):
                profit_identity_errors += 1
    if profit_identity_errors:
        _add(
            issues,
            "critical",
            "business_identity",
            f"profit != revenue - cost for {profit_identity_errors} company-year points.",
            "Derive profit after rounding revenue and cost, or use a documented numerical tolerance.",
        )

    checksum_errors: list[str] = []
    for relative_path, expected_hash in manifest.get("checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            checksum_errors.append(relative_path)
    if checksum_errors:
        _add(
            issues,
            "critical",
            "manifest_checksums",
            f"Checksum mismatch for {checksum_errors}.",
            "Do not edit generated annotations manually; rebuild and version the manifest together.",
        )

    render = manifest.get("render", {})
    expected_size = None
    if render.get("width_px") and render.get("height_px"):
        expected_size = (int(render["width_px"]), int(render["height_px"]))
    image_summary = _check_images(all_rows, issues, full_scan=full_image_scan, expected_size=expected_size)

    severity_counts = Counter(issue.severity for issue in issues)
    try:
        display_root = root.relative_to(ROOT).as_posix()
    except ValueError:
        display_root = str(root)
    report = {
        "dataset_root": display_root,
        "status": "failed" if severity_counts["critical"] or severity_counts["high"] else "passed",
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "entities": {split: len(values) for split, values in entity_sets.items()},
        "charts": {split: len(values) for split, values in chart_sets.items()},
        "unique_answer_ratio": round(unique_answer_ratio, 6),
        "task_distribution": dict(sorted(Counter(row["task_type"] for row in all_rows).items())),
        "view_distribution": dict(sorted(Counter(row["view_type"] for row in all_rows).items())),
        "difficulty_distribution": dict(sorted(Counter(row["difficulty"] for row in all_rows).items())),
        "image_summary": image_summary,
        "severity_counts": dict(severity_counts),
        "issues": [asdict(issue) for issue in issues],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate EconoChart dataset integrity and leakage.")
    parser.add_argument("--dataset-root", default="data/generated/econochart_v2")
    parser.add_argument("--full-image-scan", action="store_true")
    parser.add_argument("--report", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_dataset(args.dataset_root, full_image_scan=args.full_image_scan)
    report_path = project_path(args.report) if args.report else project_path(args.dataset_root) / "validation_report.json"
    write_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("status", "rows", "entities", "charts", "severity_counts")}, ensure_ascii=False, indent=2))
    print(f"Validation report: {report_path}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
