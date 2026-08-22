from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from econochart.data.schema import parse_ground_truth
from econochart.rewards.parsing import NUMBER_PATTERN, clauses, completion_text, extract_numbers, section_positions

RISK_ALIASES = {
    "revenue_decline": ["收入下降", "收入下滑", "收入收缩"],
    "user_churn": ["用户流失", "用户下降", "用户减少"],
    "margin_pressure": ["利润率承压", "利润率下降", "盈利空间被压缩", "盈利空间受到挤压"],
    "loss_making": ["亏损", "利润为负", "无法覆盖成本"],
    "monetization_weakness": ["变现能力", "变现效率", "单用户价值", "用户增长未充分转化"],
    "acquisition_cost_pressure": ["获客成本上升", "CAC上升", "获客压力"],
    "product_concentration": ["产品集中", "单一产品", "集中度偏高"],
    "execution_volatility": ["波动", "不稳定", "执行风险"],
}


def _ground_truths(values: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    return [parse_ground_truth(value) for value in values]


def _within_tolerance(value: float, target: dict[str, Any]) -> bool:
    expected = float(target["value"])
    tolerance = float(target.get("tolerance", max(0.2, abs(expected) * 0.005)))
    return abs(value - expected) <= tolerance


def _values_near_terms(text: str, terms: list[str]) -> list[float]:
    lowered = text.lower()
    number_matches = list(NUMBER_PATTERN.finditer(text))
    values: list[float] = []
    for raw_term in terms:
        term = str(raw_term).strip().lower()
        if not term:
            continue
        start = 0
        while (position := lowered.find(term, start)) >= 0:
            term_end = position + len(term)
            for match in number_matches:
                if match.end() < position:
                    distance = position - match.end()
                elif term_end < match.start():
                    distance = match.start() - term_end
                else:
                    distance = 0
                if distance <= 12:
                    values.append(float(match.group()))
            start = position + len(term)
    return values


def _numeric_components(
    text: str, truth: dict[str, Any]
) -> tuple[float | None, float | None, float | None]:
    targets = truth.get("numeric_targets", [])
    if not targets:
        return None, None, None
    all_candidates = extract_numbers(text)
    hits = 0.0
    for target in targets:
        terms = [str(term) for term in target.get("terms", []) if str(term).strip()]
        candidates = _values_near_terms(text, terms) if terms else all_candidates
        hits += float(any(_within_tolerance(candidate, target) for candidate in candidates))
    recall = hits / len(targets)

    support = truth.get("numeric_support", [])
    precision: float | None = None
    if support:
        if all_candidates:
            supported = sum(
                any(_within_tolerance(candidate, item) for item in support) for candidate in all_candidates
            )
            precision = supported / len(all_candidates)
        else:
            precision = 0.0
    if precision is None:
        score = recall
    elif precision + recall == 0:
        score = 0.0
    else:
        score = 2.0 * precision * recall / (precision + recall)
    return score, recall, precision


def numeric_reward(
    completions: list[Any], ground_truth: list[str | dict[str, Any]], **_: Any
) -> list[float | None]:
    return [
        _numeric_components(completion_text(completion), truth)[0]
        for completion, truth in zip(completions, _ground_truths(ground_truth))
    ]


def _trend_target_score(text: str, target: dict[str, Any]) -> float:
    metric = str(target.get("metric", ""))
    relevant = [clause for clause in clauses(text) if metric.lower() in clause.lower()]
    search_text = "。".join(relevant) if relevant else text
    has_expected = any(term.lower() in search_text.lower() for term in target.get("terms", []))
    has_opposite = any(term.lower() in search_text.lower() for term in target.get("opposites", []))
    if has_expected and not has_opposite:
        return 1.0
    if has_expected and has_opposite:
        return 0.5
    return 0.0


def trend_reward(
    completions: list[Any], ground_truth: list[str | dict[str, Any]], **_: Any
) -> list[float | None]:
    scores: list[float | None] = []
    for completion, truth in zip(completions, _ground_truths(ground_truth)):
        targets = truth.get("trend_targets", [])
        if not targets:
            scores.append(None)
            continue
        text = completion_text(completion)
        scores.append(sum(_trend_target_score(text, target) for target in targets) / len(targets))
    return scores


def format_reward(
    completions: list[Any], ground_truth: list[str | dict[str, Any]], **_: Any
) -> list[float | None]:
    scores: list[float | None] = []
    for completion, truth in zip(completions, _ground_truths(ground_truth)):
        sections = truth.get("required_sections", [])
        if not sections:
            scores.append(None)
            continue
        text = completion_text(completion)
        positions = section_positions(text, sections)
        coverage = sum(position >= 0 for position in positions) / len(sections)
        present_positions = [position for position in positions if position >= 0]
        ordered = present_positions == sorted(present_positions)
        duplicates = sum(max(0, text.count(f"【{section}】") - 1) for section in sections)
        score = 0.85 * coverage + 0.15 * float(ordered)
        score -= min(0.25, duplicates * 0.08)
        scores.append(max(0.0, min(1.0, score)))
    return scores


def evidence_reward(
    completions: list[Any], ground_truth: list[str | dict[str, Any]], **_: Any
) -> list[float | None]:
    scores: list[float | None] = []
    for completion, truth in zip(completions, _ground_truths(ground_truth)):
        keywords = truth.get("evidence_keywords", [])
        if not keywords:
            scores.append(None)
            continue
        text = completion_text(completion).lower()
        scores.append(sum(str(keyword).lower() in text for keyword in keywords) / len(keywords))
    return scores


def risk_reward(
    completions: list[Any], ground_truth: list[str | dict[str, Any]], **_: Any
) -> list[float | None]:
    scores: list[float | None] = []
    for completion, truth in zip(completions, _ground_truths(ground_truth)):
        labels = truth.get("risk_labels", [])
        if not labels:
            scores.append(None)
            continue
        text = completion_text(completion).lower()
        hits = 0
        for label in labels:
            aliases = RISK_ALIASES.get(label, [str(label)])
            hits += int(any(alias.lower() in text for alias in aliases))
        scores.append(hits / len(labels))
    return scores


def length_reward(
    completions: list[Any], ground_truth: list[str | dict[str, Any]], **_: Any
) -> list[float | None]:
    scores: list[float | None] = []
    for completion, truth in zip(completions, _ground_truths(ground_truth)):
        bounds = truth.get("length_range")
        if not bounds or len(bounds) != 2:
            scores.append(None)
            continue
        lower, upper = int(bounds[0]), int(bounds[1])
        length = len(completion_text(completion))
        if lower <= length <= upper:
            scores.append(1.0)
        elif length < lower:
            scores.append(max(0.0, length / max(1, lower)))
        else:
            scores.append(max(0.0, 1.0 - (length - upper) / max(1, upper)))
    return scores


REWARD_FUNCTIONS: list[Callable[..., list[float | None]]] = [
    numeric_reward,
    trend_reward,
    format_reward,
    evidence_reward,
    risk_reward,
    length_reward,
]


def score_completion(
    completion: Any,
    ground_truth: str | dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, float | None]:
    truth = parse_ground_truth(ground_truth)
    numeric, numeric_recall, numeric_precision = _numeric_components(completion_text(completion), truth)
    values = {
        "numeric": numeric,
        "trend": trend_reward([completion], [truth])[0],
        "format": format_reward([completion], [truth])[0],
        "evidence": evidence_reward([completion], [truth])[0],
        "risk": risk_reward([completion], [truth])[0],
        "length": length_reward([completion], [truth])[0],
    }
    component_weights = weights or {
        "numeric": 0.35,
        "trend": 0.20,
        "format": 0.15,
        "evidence": 0.15,
        "risk": 0.10,
        "length": 0.05,
    }
    applicable = [(name, value) for name, value in values.items() if value is not None]
    denominator = sum(component_weights.get(name, 0.0) for name, _ in applicable)
    overall = (
        sum(component_weights.get(name, 0.0) * float(value) for name, value in applicable) / denominator
        if denominator > 0
        else math.nan
    )
    return {
        **values,
        "numeric_recall": numeric_recall,
        "numeric_precision": numeric_precision,
        "overall": overall,
    }
