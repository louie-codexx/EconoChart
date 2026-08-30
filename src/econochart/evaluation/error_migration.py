from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from econochart.config import project_path
from econochart.data.schema import parse_ground_truth
from econochart.evaluation.metrics import normalize_answer, score_row
from econochart.io import read_jsonl, sha256_file, write_json

REFERENCE_FIELDS = ("dataset", "split", "image", "question", "answer", "ground_truth", "metadata")
METRICS = ("exact_match", "relaxed_accuracy")
STRUCTURED_MARKERS = ("【结论】", "【数据依据】", "【风险】", "【建议】")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s*%?")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
YEAR_QUESTION_PATTERN = re.compile(r"\b(?:year|when)\b|年份|哪年|何时", re.IGNORECASE)


def _to_float(value: Any) -> float | None:
    text = normalize_answer(value).replace(",", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if percent else number


def _parse_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    return [str(item).strip() for item in parsed] if isinstance(parsed, list) else [text]


def _aliases(row: dict[str, Any]) -> list[str]:
    truth = parse_ground_truth(row["ground_truth"])
    return [str(value) for value in truth.get("answer_aliases", [row["answer"]])]


def _answer_kind(row: dict[str, Any]) -> str:
    items = _parse_items(row["answer"])
    numeric = sum(_to_float(item) is not None for item in items)
    if numeric == len(items):
        return "numeric"
    if numeric:
        return "mixed"
    return "text"


def _question_type(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("question_type") or "unknown")


def _row_map(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} row {index} has no non-empty string ID")
        if record_id in mapped:
            raise ValueError(f"{label} contains duplicate ID: {record_id}")
        if "prediction" not in row:
            raise ValueError(f"{label} row has no prediction: {record_id}")
        mapped[record_id] = row
    return mapped


def _align_rows(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline = _row_map(baseline_rows, "baseline")
    candidate = _row_map(candidate_rows, "candidate")
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())[:5]
        missing_baseline = sorted(candidate.keys() - baseline.keys())[:5]
        raise ValueError(
            "Prediction IDs differ; "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    aligned = []
    for record_id in sorted(baseline):
        left, right = baseline[record_id], candidate[record_id]
        for field in REFERENCE_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(f"Reference field {field!r} differs for {record_id}")
        aligned.append((left, right))
    return aligned


def _is_binary(value: float) -> bool:
    return math.isclose(value, 0.0, abs_tol=1e-12) or math.isclose(value, 1.0, abs_tol=1e-12)


def _metric_summary(pairs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [
        (float(pair["baseline_scores"][metric]), float(pair["candidate_scores"][metric]))
        for pair in pairs
        if pair["baseline_scores"].get(metric) is not None
        and pair["candidate_scores"].get(metric) is not None
    ]
    if not values:
        return {"rows": 0}
    deltas = [right - left for left, right in values]
    summary: dict[str, Any] = {
        "rows": len(values),
        "baseline_mean": round(mean(left for left, _ in values), 6),
        "candidate_mean": round(mean(right for _, right in values), 6),
        "baseline_to_candidate_delta": round(mean(deltas), 6),
        "regressed_rows": sum(delta < -1e-12 for delta in deltas),
        "improved_rows": sum(delta > 1e-12 for delta in deltas),
        "unchanged_rows": sum(abs(delta) <= 1e-12 for delta in deltas),
        "regression_mass": round(sum(-delta for delta in deltas if delta < 0), 6),
        "improvement_mass": round(sum(delta for delta in deltas if delta > 0), 6),
    }
    if all(_is_binary(value) for pair in values for value in pair):
        transitions = Counter(
            "both_correct"
            if left == 1.0 and right == 1.0
            else "baseline_correct_candidate_wrong"
            if left == 1.0
            else "baseline_wrong_candidate_correct"
            if right == 1.0
            else "both_wrong"
            for left, right in values
        )
        summary["binary_transitions"] = {
            key: transitions[key]
            for key in (
                "both_correct",
                "baseline_correct_candidate_wrong",
                "baseline_wrong_candidate_correct",
                "both_wrong",
            )
        }
    return summary


def _behavior(row: dict[str, Any], *, max_new_tokens: int) -> dict[str, Any]:
    prediction = str(row.get("prediction", ""))
    normalized = normalize_answer(prediction)
    markers = [marker for marker in STRUCTURED_MARKERS if marker in prediction]
    completion_tokens = row.get("completion_tokens")
    cap_proxy = completion_tokens is not None and int(completion_tokens) >= max_new_tokens
    return {
        "empty": not normalized,
        "token_cap_proxy": cap_proxy,
        "structured_markers": markers,
        "has_cjk": bool(CJK_PATTERN.search(prediction)),
        "multiline": "\n" in prediction,
        "characters": len(prediction),
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "number_count": len(NUMBER_PATTERN.findall(prediction)),
    }


def _behavior_summary(rows: list[dict[str, Any]], *, max_new_tokens: int) -> dict[str, Any]:
    values = [_behavior(row, max_new_tokens=max_new_tokens) for row in rows]
    characters = [value["characters"] for value in values]
    tokens = [value["completion_tokens"] for value in values if value["completion_tokens"] is not None]
    return {
        "rows": len(rows),
        "empty_predictions": sum(value["empty"] for value in values),
        "token_cap_proxy_hits": sum(value["token_cap_proxy"] for value in values),
        "structured_marker_outputs": sum(bool(value["structured_markers"]) for value in values),
        "cjk_outputs": sum(value["has_cjk"] for value in values),
        "multiline_outputs": sum(value["multiline"] for value in values),
        "mean_characters": round(mean(characters), 3) if characters else None,
        "median_characters": round(median(characters), 3) if characters else None,
        "mean_completion_tokens": round(mean(tokens), 3) if tokens else None,
        "median_completion_tokens": round(median(tokens), 3) if tokens else None,
    }


def _reference_embedded(row: dict[str, Any], prediction: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    for alias in _aliases(row):
        normalized_alias = normalize_answer(alias)
        if not normalized_alias:
            continue
        if _to_float(alias) is None:
            if len(normalized_alias) >= 2 and normalized_alias in normalized_prediction:
                return True
            continue
        target = _to_float(alias)
        for match in NUMBER_PATTERN.findall(prediction):
            value = _to_float(match)
            if target is not None and value is not None and math.isclose(target, value, rel_tol=0, abs_tol=1e-12):
                return True
    return False


def _single_numeric_target(row: dict[str, Any]) -> float | None:
    items = _parse_items(row["answer"])
    if len(items) != 1:
        return None
    return _to_float(items[0])


def _is_year_target(row: dict[str, Any]) -> bool:
    target = _single_numeric_target(row)
    if target is None or not target.is_integer() or not 1800 <= target <= 2200:
        return False
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    year_flags = [str(value).upper() for value in metadata.get("year_flags", [])]
    return bool(YEAR_QUESTION_PATTERN.search(str(row.get("question", "")))) or "YES" in year_flags


def _single_numeric_token(value: Any) -> float | None:
    matches = NUMBER_PATTERN.findall(str(value))
    if len(matches) != 1:
        return None
    return _to_float(matches[0])


def _percent_scale_ambiguity(row: dict[str, Any], prediction: str) -> bool:
    target = _single_numeric_target(row)
    matches = NUMBER_PATTERN.findall(prediction)
    if target is None or len(matches) != 1 or "%" not in matches[0]:
        return False
    raw_text = normalize_answer(matches[0]).replace(",", "").replace("%", "").strip()
    try:
        raw_value = float(raw_text)
    except ValueError:
        return False
    return math.isclose(raw_value, target, rel_tol=0, abs_tol=1e-12)


def _numeric_relative_error(target: float, prediction: float) -> float:
    if target == 0:
        return 0.0 if prediction == 0 else math.inf
    return abs(prediction - target) / abs(target)


def _numeric_error_band(error: float) -> str:
    if error <= 0.05:
        return "within_5pct"
    if error <= 0.10:
        return "5_to_10pct"
    return "over_10pct"


def _proxy_reason(pair: dict[str, Any], *, max_new_tokens: int) -> str:
    row = pair["baseline"]
    baseline_prediction = str(pair["baseline"].get("prediction", ""))
    candidate_prediction = str(pair["candidate"].get("prediction", ""))
    baseline_behavior = _behavior(pair["baseline"], max_new_tokens=max_new_tokens)
    candidate_behavior = _behavior(pair["candidate"], max_new_tokens=max_new_tokens)
    if candidate_behavior["empty"]:
        return "empty_prediction"
    if candidate_behavior["token_cap_proxy"] and not baseline_behavior["token_cap_proxy"]:
        return "new_token_cap_proxy_hit"
    if candidate_behavior["structured_markers"] and not baseline_behavior["structured_markers"]:
        return "structured_template_intrusion"
    kind = _answer_kind(row)
    if _reference_embedded(row, candidate_prediction):
        if kind == "numeric":
            return "numeric_reference_embedded_with_extra_output"
        if kind == "text":
            return "text_reference_embedded_with_extra_output"
        return "reference_embedded_with_extra_output"
    if kind == "numeric":
        if _is_year_target(row):
            return "year_value_or_answer_type_error"
        if _percent_scale_ambiguity(row, candidate_prediction):
            return "numeric_percent_scale_ambiguity"
        target = _single_numeric_target(row)
        candidate_value = _to_float(candidate_prediction)
        if target is not None and candidate_value is not None:
            error = _numeric_relative_error(target, candidate_value)
            return f"numeric_value_error_{_numeric_error_band(error)}"
        embedded_value = _single_numeric_token(candidate_prediction)
        if target is not None and embedded_value is not None:
            error = _numeric_relative_error(target, embedded_value)
            return f"numeric_unit_or_text_output_{_numeric_error_band(error)}"
        if NUMBER_PATTERN.search(candidate_prediction):
            return "numeric_value_embedded_or_multi_number_error"
        return "numeric_missing_or_non_numeric"
    if kind == "mixed" or len(_parse_items(row["answer"])) > 1:
        return "multi_answer_structure_or_content_error"
    if normalize_answer(baseline_prediction) != normalize_answer(candidate_prediction):
        return "text_label_or_content_error"
    return "unclassified_score_change"


def _group_summaries(pairs: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        row = pair["baseline"]
        if field == "answer_kind":
            value = _answer_kind(row)
        elif field == "question_type":
            value = _question_type(row)
        else:
            value = str(row.get(field) or "unknown")
        buckets[value].append(pair)
    return {
        value: {
            "rows": len(bucket),
            "metrics": {metric: _metric_summary(bucket, metric) for metric in METRICS},
        }
        for value, bucket in sorted(buckets.items())
    }


def _regression_examples(
    pairs: list[dict[str, Any]], *, max_new_tokens: int, examples_per_reason: int
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    regressed = [
        pair
        for pair in pairs
        if float(pair["baseline_scores"]["exact_match"])
        > float(pair["candidate_scores"]["exact_match"])
    ]
    reasons = Counter(_proxy_reason(pair, max_new_tokens=max_new_tokens) for pair in regressed)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered = sorted(
        regressed,
        key=lambda pair: (
            float(pair["candidate_scores"]["relaxed_accuracy"])
            - float(pair["baseline_scores"]["relaxed_accuracy"]),
            pair["baseline"]["id"],
        ),
    )
    for pair in ordered:
        reason = _proxy_reason(pair, max_new_tokens=max_new_tokens)
        if len(examples[reason]) >= examples_per_reason:
            continue
        row = pair["baseline"]
        examples[reason].append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "answer_kind": _answer_kind(row),
                "question_type": _question_type(row),
                "year_target": _is_year_target(row),
                "image": row.get("image"),
                "question": row["question"],
                "answer": row["answer"],
                "baseline_prediction": pair["baseline"].get("prediction", ""),
                "candidate_prediction": pair["candidate"].get("prediction", ""),
                "baseline_completion_tokens": pair["baseline"].get("completion_tokens"),
                "candidate_completion_tokens": pair["candidate"].get("completion_tokens"),
                "baseline_scores": pair["baseline_scores"],
                "candidate_scores": pair["candidate_scores"],
            }
        )
    return dict(sorted(reasons.items())), dict(sorted(examples.items()))


def _exact_regression_metric_scopes(
    pairs: list[dict[str, Any]], *, max_new_tokens: int
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    scopes: Counter[str] = Counter()
    by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        if float(pair["baseline_scores"]["exact_match"]) <= float(
            pair["candidate_scores"]["exact_match"]
        ):
            continue
        relaxed_delta = float(pair["candidate_scores"]["relaxed_accuracy"]) - float(
            pair["baseline_scores"]["relaxed_accuracy"]
        )
        if relaxed_delta < -1e-12:
            scope = "exact_and_relaxed_regression"
        elif relaxed_delta > 1e-12:
            scope = "exact_regression_relaxed_improved"
        else:
            scope = "exact_only_relaxed_unchanged"
        reason = _proxy_reason(pair, max_new_tokens=max_new_tokens)
        scopes[scope] += 1
        by_reason[reason][scope] += 1
    return (
        dict(sorted(scopes.items())),
        {reason: dict(sorted(values.items())) for reason, values in sorted(by_reason.items())},
    )


def _regression_signatures(pairs: list[dict[str, Any]], *, max_new_tokens: int) -> dict[str, int]:
    signatures: Counter[str] = Counter()
    for pair in pairs:
        if float(pair["baseline_scores"]["exact_match"]) <= float(
            pair["candidate_scores"]["exact_match"]
        ):
            continue
        row = pair["baseline"]
        baseline_prediction = str(pair["baseline"].get("prediction", ""))
        candidate_prediction = str(pair["candidate"].get("prediction", ""))
        baseline_behavior = _behavior(pair["baseline"], max_new_tokens=max_new_tokens)
        candidate_behavior = _behavior(pair["candidate"], max_new_tokens=max_new_tokens)
        signatures[f"answer_kind_{_answer_kind(row)}"] += 1
        if candidate_behavior["empty"]:
            signatures["candidate_empty"] += 1
        if candidate_behavior["token_cap_proxy"]:
            signatures["candidate_token_cap_proxy"] += 1
        if candidate_behavior["token_cap_proxy"] and not baseline_behavior["token_cap_proxy"]:
            signatures["new_token_cap_proxy"] += 1
        if candidate_behavior["structured_markers"]:
            signatures["candidate_structured_template"] += 1
        if candidate_behavior["structured_markers"] and not baseline_behavior["structured_markers"]:
            signatures["new_structured_template"] += 1
        if candidate_behavior["has_cjk"] and not baseline_behavior["has_cjk"]:
            signatures["new_cjk_output"] += 1
        if candidate_behavior["multiline"] and not baseline_behavior["multiline"]:
            signatures["new_multiline_output"] += 1
        if _reference_embedded(row, candidate_prediction):
            signatures["reference_embedded_in_wrong_output"] += 1
        if len(candidate_prediction) > max(20, 2 * len(baseline_prediction)):
            signatures["candidate_more_than_2x_baseline_chars"] += 1
    return dict(sorted(signatures.items()))


def _assert_expected(report: dict[str, Any], expected: dict[str, float | int | None]) -> None:
    if expected.get("paired_rows") is not None and report["inputs"]["paired_rows"] != expected["paired_rows"]:
        raise ValueError(
            f"Expected {expected['paired_rows']} paired rows, got {report['inputs']['paired_rows']}"
        )
    paths = {
        "baseline_exact": ("exact_match", "baseline_mean"),
        "candidate_exact": ("exact_match", "candidate_mean"),
        "baseline_relaxed": ("relaxed_accuracy", "baseline_mean"),
        "candidate_relaxed": ("relaxed_accuracy", "candidate_mean"),
    }
    for name, (metric, key) in paths.items():
        target = expected.get(name)
        if target is None:
            continue
        actual = report["overall"][metric][key]
        if round(float(actual), 6) != round(float(target), 6):
            raise ValueError(f"Expected {name}={target:.6f}, got {actual:.6f}")


def analyze_error_migration(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = 128,
    examples_per_reason: int = 5,
) -> dict[str, Any]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if examples_per_reason < 0:
        raise ValueError("examples_per_reason must be non-negative")
    aligned = _align_rows(baseline_rows, candidate_rows)
    pairs = []
    for baseline, candidate in aligned:
        pairs.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "baseline_scores": score_row(baseline),
                "candidate_scores": score_row(candidate),
            }
        )
    reason_counts, examples = _regression_examples(
        pairs,
        max_new_tokens=max_new_tokens,
        examples_per_reason=examples_per_reason,
    )
    metric_scope_counts, reason_metric_scopes = _exact_regression_metric_scopes(
        pairs, max_new_tokens=max_new_tokens
    )
    dataset_counts = Counter(str(pair["baseline"]["dataset"]) for pair in pairs)
    return {
        "schema_version": 2,
        "inputs": {
            "paired_rows": len(pairs),
            "dataset_counts": dict(sorted(dataset_counts.items())),
            "reference_fields_verified": list(REFERENCE_FIELDS),
            "max_new_tokens_for_cap_proxy": max_new_tokens,
        },
        "overall": {metric: _metric_summary(pairs, metric) for metric in METRICS},
        "breakdowns": {
            field: _group_summaries(pairs, field)
            for field in ("dataset", "answer_kind", "question_type")
        },
        "output_behavior": {
            "baseline": _behavior_summary(baseline_rows, max_new_tokens=max_new_tokens),
            "candidate": _behavior_summary(candidate_rows, max_new_tokens=max_new_tokens),
        },
        "exact_regression_proxy_attribution": {
            "notice": (
                "Descriptive proxies only. Year, unit, percent, and embedded-reference categories remain "
                "heuristic and do not by themselves prove semantic correctness, OCR failure, or domain "
                "overfitting. Primary registered metrics are not changed by this diagnostic."
            ),
            "primary_reason_counts": reason_counts,
            "metric_scope_counts": metric_scope_counts,
            "primary_reason_by_metric_scope": reason_metric_scopes,
            "nonexclusive_signature_counts": _regression_signatures(
                pairs, max_new_tokens=max_new_tokens
            ),
            "examples": examples,
        },
    }


def _print_summary(report: dict[str, Any], output: str) -> None:
    inputs = report["inputs"]
    print(
        "ERROR_MIGRATION_INPUT_GATE=PASS "
        f"paired_rows={inputs['paired_rows']} datasets={json.dumps(inputs['dataset_counts'], sort_keys=True)}"
    )
    for metric in METRICS:
        summary = report["overall"][metric]
        print(
            f"METRIC[overall.{metric}]="
            f"base:{summary['baseline_mean']:.6f} sft:{summary['candidate_mean']:.6f} "
            f"delta:{summary['baseline_to_candidate_delta']:+.6f} "
            f"regressed:{summary['regressed_rows']} improved:{summary['improved_rows']}"
        )
        if "binary_transitions" in summary:
            print(f"TRANSITIONS[overall.{metric}]={json.dumps(summary['binary_transitions'], sort_keys=True)}")
    for dataset, block in report["breakdowns"]["dataset"].items():
        for metric in METRICS:
            summary = block["metrics"][metric]
            print(
                f"METRIC[dataset.{dataset}.{metric}]="
                f"base:{summary['baseline_mean']:.6f} sft:{summary['candidate_mean']:.6f} "
                f"delta:{summary['baseline_to_candidate_delta']:+.6f}"
            )
    for label in ("baseline", "candidate"):
        behavior = report["output_behavior"][label]
        print(f"OUTPUT_BEHAVIOR[{label}]={json.dumps(behavior, sort_keys=True)}")
    print(
        "EXACT_REGRESSION_PRIMARY_PROXIES="
        + json.dumps(
            report["exact_regression_proxy_attribution"]["primary_reason_counts"], sort_keys=True
        )
    )
    print(
        "EXACT_REGRESSION_SIGNATURES="
        + json.dumps(
            report["exact_regression_proxy_attribution"]["nonexclusive_signature_counts"],
            sort_keys=True,
        )
    )
    print(
        "EXACT_REGRESSION_METRIC_SCOPES="
        + json.dumps(
            report["exact_regression_proxy_attribution"]["metric_scope_counts"], sort_keys=True
        )
    )
    print(
        "EXACT_REGRESSION_REASON_BY_SCOPE="
        + json.dumps(
            report["exact_regression_proxy_attribution"]["primary_reason_by_metric_scope"],
            sort_keys=True,
        )
    )
    for kind, block in report["breakdowns"]["answer_kind"].items():
        exact = block["metrics"]["exact_match"]
        relaxed = block["metrics"]["relaxed_accuracy"]
        print(
            f"ANSWER_KIND[{kind}]=rows:{block['rows']} "
            f"exact_delta:{exact['baseline_to_candidate_delta']:+.6f} "
            f"relaxed_delta:{relaxed['baseline_to_candidate_delta']:+.6f}"
        )
    for question_type, block in report["breakdowns"]["question_type"].items():
        exact = block["metrics"]["exact_match"]
        relaxed = block["metrics"]["relaxed_accuracy"]
        print(
            f"QUESTION_TYPE[{question_type}]=rows:{block['rows']} "
            f"exact_delta:{exact['baseline_to_candidate_delta']:+.6f} "
            f"relaxed_delta:{relaxed['baseline_to_candidate_delta']:+.6f}"
        )
    print(f"ERROR_MIGRATION_REPORT={output}")
    print("ERROR_MIGRATION_GATE=PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only paired Base-to-SFT error migration diagnostics for fixed external predictions."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--examples-per-reason", type=int, default=5)
    parser.add_argument("--expected-paired-rows", type=int)
    parser.add_argument("--expected-baseline-exact", type=float)
    parser.add_argument("--expected-candidate-exact", type=float)
    parser.add_argument("--expected-baseline-relaxed", type=float)
    parser.add_argument("--expected-candidate-relaxed", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_path = project_path(args.baseline)
    candidate_path = project_path(args.candidate)
    report = analyze_error_migration(
        list(read_jsonl(baseline_path)),
        list(read_jsonl(candidate_path)),
        max_new_tokens=args.max_new_tokens,
        examples_per_reason=args.examples_per_reason,
    )
    report["inputs"].update(
        {
            "baseline_path": str(baseline_path),
            "candidate_path": str(candidate_path),
            "baseline_sha256": sha256_file(baseline_path),
            "candidate_sha256": sha256_file(candidate_path),
        }
    )
    _assert_expected(
        report,
        {
            "paired_rows": args.expected_paired_rows,
            "baseline_exact": args.expected_baseline_exact,
            "candidate_exact": args.expected_candidate_exact,
            "baseline_relaxed": args.expected_baseline_relaxed,
            "candidate_relaxed": args.expected_candidate_relaxed,
        },
    )
    output_path = project_path(args.output)
    write_json(output_path, report)
    _print_summary(report, str(output_path))


if __name__ == "__main__":
    main()
