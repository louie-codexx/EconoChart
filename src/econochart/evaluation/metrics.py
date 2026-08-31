from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean, median
from typing import Any

from econochart.data.schema import parse_ground_truth
from econochart.rewards.components import score_completion


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower().strip(".\n")
    return re.sub(r"\s+", " ", text)


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


def relaxed_accuracy(prediction: str, answers: Iterable[str], tolerance: float = 0.05) -> float:
    prediction_number = _to_float(prediction)
    for answer in answers:
        answer_number = _to_float(answer)
        if prediction_number is not None and answer_number is not None:
            if answer_number == 0 and prediction_number == 0:
                return 1.0
            if answer_number != 0 and abs(prediction_number - answer_number) / abs(answer_number) <= tolerance:
                return 1.0
        elif normalize_answer(prediction) == normalize_answer(answer):
            return 1.0
    return 0.0


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(target: str, prediction: str, threshold: float = 0.5) -> float:
    target_text = normalize_answer(target)
    prediction_text = normalize_answer(prediction)
    denominator = max(len(target_text), len(prediction_text))
    similarity = 1.0 if denominator == 0 else 1.0 - _edit_distance(target_text, prediction_text) / denominator
    return similarity if similarity >= threshold else 0.0


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    return [str(item).strip() for item in parsed] if isinstance(parsed, list) else [text]


def chartqapro_accuracy(record: dict[str, Any], prediction: str) -> float:
    targets = _parse_list(record["answer"])
    predictions = _parse_list(prediction)
    flags = [str(value).upper() for value in record.get("metadata", {}).get("year_flags", ["NO"])]
    flags = flags[-len(targets) :]
    if len(flags) < len(targets):
        flags = (flags * len(targets))[: len(targets)]
    exact_question = record.get("metadata", {}).get("question_type") in {"Fact Checking", "Multi Choice"}
    scores = []
    for index in range(max(len(targets), len(predictions))):
        if index >= len(targets) or index >= len(predictions):
            scores.append(0.0)
            continue
        target, predicted = targets[index], predictions[index]
        if exact_question or (index < len(flags) and flags[index] == "YES"):
            scores.append(float(normalize_answer(target) == normalize_answer(predicted)))
            continue
        target_number, prediction_number = _to_float(target), _to_float(predicted)
        if target_number is not None and prediction_number is not None:
            scores.append(relaxed_accuracy(predicted, [target]))
        else:
            scores.append(anls(target, predicted))
    return mean(scores) if scores else 0.0


_MMEFINANCE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._/%+-][a-z0-9]+)*")
_MMEFINANCE_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


def _mmefinance_token_f1(reference: str, prediction: str) -> float:
    reference_tokens = _MMEFINANCE_TOKEN_PATTERN.findall(normalize_answer(reference))
    prediction_tokens = _MMEFINANCE_TOKEN_PATTERN.findall(normalize_answer(prediction))
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _mmefinance_numbers(value: str) -> list[float]:
    numbers = []
    for token in _MMEFINANCE_NUMBER_PATTERN.findall(value):
        parsed = _to_float(token)
        if parsed is not None and math.isfinite(parsed):
            numbers.append(parsed)
    return numbers


def _numbers_match(reference: float, prediction: float, tolerance: float = 0.05) -> bool:
    if reference == 0:
        return prediction == 0
    return abs(prediction - reference) / abs(reference) <= tolerance


def _mmefinance_numeric_precision_recall(
    reference: str,
    prediction: str,
) -> tuple[float | None, float | None]:
    reference_numbers = _mmefinance_numbers(reference)
    if not reference_numbers:
        return None, None
    prediction_numbers = _mmefinance_numbers(prediction)
    if not prediction_numbers:
        return 0.0, 0.0
    unmatched = set(range(len(prediction_numbers)))
    matches = 0
    for reference_number in reference_numbers:
        candidates = [
            index
            for index in unmatched
            if _numbers_match(reference_number, prediction_numbers[index])
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(reference_number - prediction_numbers[index]))
        unmatched.remove(best)
        matches += 1
    return matches / len(prediction_numbers), matches / len(reference_numbers)


def mmefinance_surrogate_scores(reference: str, prediction: str) -> dict[str, float | None]:
    numeric_precision, numeric_recall = _mmefinance_numeric_precision_recall(
        reference,
        prediction,
    )
    return {
        "surrogate_exact_match": float(normalize_answer(reference) == normalize_answer(prediction)),
        "surrogate_anls": anls(reference, prediction),
        "surrogate_token_f1": _mmefinance_token_f1(reference, prediction),
        "surrogate_numeric_precision": numeric_precision,
        "surrogate_numeric_recall": numeric_recall,
        "output_nonempty": float(bool(prediction.strip())),
    }


def score_row(row: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, float | None]:
    dataset = str(row["dataset"]).lower()
    prediction = str(row.get("prediction", ""))
    if dataset.startswith("econochart-v2"):
        return score_completion(prediction, row["ground_truth"], weights)
    truth = parse_ground_truth(row["ground_truth"])
    aliases = [str(value) for value in truth.get("answer_aliases", [row["answer"]])]
    if dataset == "mmefinance":
        return mmefinance_surrogate_scores(str(row["answer"]), prediction)
    if dataset == "chartqapro":
        return {
            "relaxed_accuracy": chartqapro_accuracy(row, prediction),
            "exact_match": float(normalize_answer(prediction) in {normalize_answer(value) for value in aliases}),
        }
    return {
        "relaxed_accuracy": relaxed_accuracy(prediction, aliases),
        "exact_match": float(normalize_answer(prediction) in {normalize_answer(value) for value in aliases}),
    }


def _average_metrics(scored: list[dict[str, float | None]]) -> dict[str, float]:
    names = sorted({name for row in scored for name in row})
    result = {}
    for name in names:
        values = [float(row[name]) for row in scored if row.get(name) is not None and math.isfinite(float(row[name]))]
        if values:
            result[name] = round(mean(values), 6)
    return result


def _slice_value(row: dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is None and field in {"scenario", "task_category", "image_type", "image_style"}:
        value = row.get("metadata", {}).get(field)
    return value


def _group(rows: list[dict[str, Any]], scored: list[dict[str, float | None]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for row, score in zip(rows, scored):
        value = _slice_value(row, field)
        buckets[str(value or "unknown")].append(score)
    return {
        key: {"rows": len(values), "metrics": _average_metrics(values)}
        for key, values in sorted(buckets.items())
    }


def aggregate_metrics(rows: list[dict[str, Any]], weights: dict[str, float] | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction list")
    scored = [score_row(row, weights) for row in rows]
    latency = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    completion_tokens = [int(row["completion_tokens"]) for row in rows if row.get("completion_tokens") is not None]
    efficiency: dict[str, Any] = {}
    if latency:
        efficiency.update(
            {
                "mean_latency_seconds": round(mean(latency), 6),
                "median_latency_seconds": round(median(latency), 6),
                "samples_per_second": round(len(latency) / sum(latency), 6) if sum(latency) else None,
            }
        )
    if completion_tokens:
        efficiency["mean_completion_tokens"] = round(mean(completion_tokens), 3)
    report = {
        "rows": len(rows),
        "overall": _average_metrics(scored),
        "slices": {
            field: _group(rows, scored, field)
            for field in (
                "dataset",
                "task_type",
                "view_type",
                "industry",
                "difficulty",
                "scenario",
                "task_category",
                "image_type",
                "image_style",
            )
        },
        "efficiency": efficiency,
    }
    if any(str(row.get("dataset", "")).lower() == "mmefinance" for row in rows):
        report["score_contract"] = {
            "official": "Use the upstream MME-Finance image-aware judge for the official benchmark score.",
            "local": (
                "surrogate_* metrics are deterministic audit diagnostics only and must not be "
                "reported as the official MME-Finance score."
            ),
        }
    return report


def chartqapro_official_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported = []
    for row in rows:
        if str(row.get("dataset", "")).lower() != "chartqapro":
            continue
        metadata = row.get("metadata", {})
        exported.append(
            {
                "Answer": metadata.get("answer_sequence", [row["answer"]]),
                "Question Type": metadata.get("question_type", "unknown"),
                "Year": metadata.get("year_flags", ["NO"]),
                "prediction": row.get("prediction", ""),
            }
        )
    return exported


def mmefinance_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported = []
    for row in rows:
        if str(row.get("dataset", "")).lower() != "mmefinance":
            continue
        metadata = row.get("metadata", {})
        exported.append(
            {
                "index": metadata.get("source_index"),
                "image": row.get("image"),
                "source_image_path": metadata.get("source_image_path"),
                "image_type": metadata.get("image_type"),
                "image_style": metadata.get("image_style"),
                "task_category": metadata.get("task_category"),
                "question": row.get("question"),
                "reference_answer": row.get("answer"),
                "background": metadata.get("background", ""),
                "prediction": row.get("prediction", ""),
                "surrogate_scores": mmefinance_surrogate_scores(
                    str(row.get("answer", "")),
                    str(row.get("prediction", "")),
                ),
            }
        )
    return exported
