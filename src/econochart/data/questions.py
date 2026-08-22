from __future__ import annotations

import math
import random
from typing import Any

from econochart.data.generator import percentage_change
from econochart.data.schema import DATASET_VERSION, make_record
from econochart.io import stable_seed

RISK_TEXT = {
    "revenue_decline": "收入规模下降",
    "user_churn": "活跃用户流失",
    "margin_pressure": "利润率承压",
    "loss_making": "出现经营亏损",
    "monetization_weakness": "用户增长未充分转化为收入",
    "acquisition_cost_pressure": "获客成本上升",
    "product_concentration": "产品收入集中度偏高",
    "execution_volatility": "增长与效率仍存在波动",
}

RECOMMENDATION_TEXT = {
    "revenue_decline": "优先修复核心产品需求并按渠道拆解收入流失来源",
    "user_churn": "开展用户分层召回并定位留存下降的关键环节",
    "margin_pressure": "压降低回报投入并建立毛利率和费用率双重预警",
    "loss_making": "收缩低贡献业务并设定分阶段盈亏平衡目标",
    "monetization_weakness": "优化付费转化与产品分层定价，提升单用户价值",
    "acquisition_cost_pressure": "将投放预算向高生命周期价值渠道迁移",
    "product_concentration": "培育第二增长曲线并降低对单一产品的依赖",
    "execution_volatility": "按月监控收入、用户和利润率的偏差并及时校正",
}

TREND_TERMS = {
    "increase": {"terms": ["增长", "上升", "扩大"], "opposites": ["下降", "减少", "收缩"]},
    "decrease": {"terms": ["下降", "减少", "收缩"], "opposites": ["增长", "上升", "扩大"]},
    "stable": {"terms": ["稳定", "平稳", "基本持平"], "opposites": ["明显增长", "明显下降"]},
}


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _answer(
    conclusion: str,
    evidence: str,
    *,
    risk: str | None = None,
    recommendation: str | None = None,
) -> tuple[str, list[str]]:
    sections = ["结论", "数据依据"]
    parts = [f"【结论】{conclusion}", f"【数据依据】{evidence}"]
    if risk:
        sections.append("风险")
        parts.append(f"【风险】{risk}")
    if recommendation:
        sections.append("建议")
        parts.append(f"【建议】{recommendation}")
    return "\n".join(parts), sections


def _default_numeric_terms(name: str) -> list[str]:
    normalized = name.lower()
    if "growth_gap" in normalized:
        return ["增速差"]
    if "cagr" in normalized:
        return ["CAGR", "复合增长率"]
    terms: list[str]
    if "margin" in normalized:
        terms = ["利润率", "Margin"]
    elif "conversion" in normalized:
        terms = ["转化率", "Conversion"]
    elif "cac" in normalized:
        terms = ["CAC", "获客成本"]
    elif "arpu" in normalized:
        terms = ["ARPU", "单用户"]
    elif "user" in normalized:
        terms = ["用户", "User"]
    elif "cost" in normalized:
        terms = ["成本", "Cost"]
    elif "profit" in normalized:
        terms = ["利润", "Profit"]
    elif "revenue" in normalized:
        terms = ["收入", "Revenue"]
    else:
        terms = []
    if "change" in normalized:
        terms.append("变化")
    if "yoy" in normalized:
        terms.append("同比")
    return terms


def _numeric(
    name: str,
    value: float,
    unit: str,
    *,
    tolerance: float | None = None,
    terms: list[str] | None = None,
) -> dict[str, Any]:
    if tolerance is None:
        tolerance = max(0.2, abs(value) * 0.005)
    return {
        "name": name,
        "value": round(float(value), 4),
        "unit": unit,
        "tolerance": round(tolerance, 4),
        "terms": terms if terms is not None else _default_numeric_terms(name),
    }


def _year_target(name: str, value: int, *, terms: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": "year",
        "tolerance": 0.0,
        "terms": terms or ["年"],
    }


def _numeric_support(company: dict[str, Any], targets: list[dict[str, Any]]) -> list[dict[str, float]]:
    support: dict[float, float] = {}

    def add(value: Any, tolerance: float | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        numeric = round(float(value), 4)
        if not math.isfinite(numeric):
            return
        if tolerance is None:
            tolerance = 0.0 if numeric.is_integer() and 2000 <= numeric <= 2100 else max(0.2, abs(numeric) * 0.005)
        support[numeric] = max(support.get(numeric, 0.0), round(float(tolerance), 4))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        else:
            add(value)

    for field in ("years", "metrics", "facts", "product_mix_pct", "region_mix_pct"):
        visit(company.get(field))
    for target in targets:
        add(target.get("value"), target.get("tolerance"))
    return [{"value": value, "tolerance": support[value]} for value in sorted(support)]


def _trend(metric: str, label: str) -> dict[str, Any]:
    terms = TREND_TERMS[label]
    return {"metric": metric, "label": label, "terms": terms["terms"], "opposites": terms["opposites"]}


def _trend_phrase(metric: str, label: str) -> str:
    direction = {"increase": "增长", "decrease": "下降", "stable": "基本持平"}[label]
    return f"{metric}整体{direction}"


def _risk_sentence(company: dict[str, Any], limit: int = 2) -> str:
    return "；".join(RISK_TEXT[code] for code in company["facts"]["risks"][:limit]) + "。"


def _recommendation_sentence(company: dict[str, Any], limit: int = 2) -> str:
    return "；".join(RECOMMENDATION_TEXT[code] for code in company["facts"]["risks"][:limit]) + "。"


def _base_ground_truth(
    sections: list[str],
    *,
    numeric_targets: list[dict[str, Any]] | None = None,
    trend_targets: list[dict[str, Any]] | None = None,
    evidence_keywords: list[str] | None = None,
    risk_labels: list[str] | None = None,
    grpo_eligible: bool = True,
    min_chars: int = 45,
    max_chars: int = 420,
) -> dict[str, Any]:
    return {
        "required_sections": sections,
        "numeric_targets": numeric_targets or [],
        "trend_targets": trend_targets or [],
        "evidence_keywords": evidence_keywords or [],
        "risk_labels": risk_labels or [],
        "grpo_eligible": grpo_eligible,
        "length_range": [min_chars, max_chars],
    }


def _record(
    company: dict[str, Any],
    view_type: str,
    image_path: str,
    suffix: str,
    task_type: str,
    difficulty: str,
    question: str,
    answer: str,
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    chart_id = f"{company['entity_id']}_{view_type}"
    ground_truth = dict(ground_truth)
    ground_truth["numeric_support"] = _numeric_support(company, ground_truth.get("numeric_targets", []))
    return make_record(
        record_id=f"{chart_id}_{suffix}",
        dataset=DATASET_VERSION,
        split=company["split"],
        entity_id=company["entity_id"],
        chart_id=chart_id,
        image=image_path,
        industry=company["industry"],
        view_type=view_type,
        task_type=task_type,
        difficulty=difficulty,
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        metadata={"industry_zh": company["industry_zh"], "scenario": company["scenario"]},
    )


def _financial_records(company: dict[str, Any], image_path: str) -> list[dict[str, Any]]:
    metrics = company["metrics"]
    facts = company["facts"]
    revenue, cost, profit = metrics["revenue_million"], metrics["cost_million"], metrics["profit_million"]
    margin = metrics["profit_margin_pct"]
    industry = company["industry_zh"]
    records: list[dict[str, Any]] = []

    answer, sections = _answer(
        f"2025年营业收入为{_fmt(revenue[-1])}百万元，利润为{_fmt(profit[-1])}百万元。",
        f"图中2025年Revenue={_fmt(revenue[-1])}，Cost={_fmt(cost[-1])}，两者相减得到Profit={_fmt(profit[-1])}百万元。",
    )
    records.append(
        _record(
            company,
            "financial",
            image_path,
            "current_value",
            "value_retrieval",
            "easy",
            f"这家{industry}企业2025年的营业收入和利润分别是多少？请给出图中数值与单位。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_2025", revenue[-1], "百万元"),
                    _numeric("profit_2025", profit[-1], "百万元"),
                ],
                evidence_keywords=["收入", "利润"],
                max_chars=220,
            ),
        )
    )

    answer, sections = _answer(
        f"2021—2025年收入累计变化{_fmt(facts['revenue_change_pct'])}%，年复合增长率约{_fmt(facts['revenue_cagr_pct'])}%。",
        f"收入由{_fmt(revenue[0])}百万元变为{_fmt(revenue[-1])}百万元；同期利润率由{_fmt(margin[0])}%变为{_fmt(margin[-1])}%，变化{_fmt(facts['margin_change_pp'])}个百分点。",
    )
    records.append(
        _record(
            company,
            "financial",
            image_path,
            "growth_math",
            "numerical_reasoning",
            "medium",
            "计算2021—2025年营业收入累计变化率和年复合增长率，并说明同期利润率变化。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("revenue_cagr_pct", facts["revenue_cagr_pct"], "%", tolerance=0.3),
                    _numeric("margin_change_pp", facts["margin_change_pp"], "个百分点", tolerance=0.3),
                ],
                evidence_keywords=["收入", "复合增长率", "利润率"],
                max_chars=280,
            ),
        )
    )

    relation = (
        "收入增速高于成本增速，盈利空间总体改善"
        if facts["revenue_change_pct"] > facts["cost_change_pct"] + 3.0
        else "成本增速高于收入增速，盈利空间受到挤压"
        if facts["cost_change_pct"] > facts["revenue_change_pct"] + 3.0
        else "收入与成本增速接近，盈利空间整体平稳"
    )
    answer, sections = _answer(
        f"{relation}。",
        f"收入累计变化{_fmt(facts['revenue_change_pct'])}%；成本累计变化{_fmt(facts['cost_change_pct'])}%；"
        f"{_trend_phrase('利润率', facts['margin_trend'])}，变化{_fmt(facts['margin_change_pp'])}个百分点。",
        risk=_risk_sentence(company, 1),
    )
    records.append(
        _record(
            company,
            "financial",
            image_path,
            "relationship",
            "relationship_analysis",
            "hard",
            "分析收入、成本与利润之间的关系，并用至少两项图中数据支持判断。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("cost_change_pct", facts["cost_change_pct"], "%", tolerance=0.3),
                    _numeric("margin_change_pp", facts["margin_change_pp"], "个百分点", tolerance=0.3),
                ],
                trend_targets=[_trend("利润率", facts["margin_trend"])],
                evidence_keywords=["收入", "成本", "利润率"],
                risk_labels=company["facts"]["risks"][:1],
            ),
        )
    )

    answer, sections = _answer(
        "企业需要在增长质量与盈利能力之间取得平衡。",
        f"2021—2025年收入变化{_fmt(facts['revenue_change_pct'])}%，用户变化{_fmt(facts['user_change_pct'])}%，利润率变化{_fmt(facts['margin_change_pp'])}个百分点。",
        risk=_risk_sentence(company),
        recommendation=_recommendation_sentence(company),
    )
    records.append(
        _record(
            company,
            "financial",
            image_path,
            "decision",
            "decision_support",
            "hard",
            f"请为这家{industry}企业形成简短经营诊断：给出结论、数据依据、主要风险和可执行建议。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("user_change_pct", facts["user_change_pct"], "%", tolerance=0.3),
                    _numeric("margin_change_pp", facts["margin_change_pp"], "个百分点", tolerance=0.3),
                ],
                evidence_keywords=["收入", "用户", "利润率"],
                risk_labels=company["facts"]["risks"][:2],
                grpo_eligible=False,
                min_chars=100,
                max_chars=520,
            ),
        )
    )
    return records


def _operations_records(company: dict[str, Any], image_path: str) -> list[dict[str, Any]]:
    metrics, facts = company["metrics"], company["facts"]
    users, arpu = metrics["active_users_10k"], metrics["arpu_yuan"]
    conversion, cac = metrics["conversion_rate_pct"], metrics["cac_yuan"]
    records: list[dict[str, Any]] = []

    answer, sections = _answer(
        f"2025年活跃用户为{_fmt(users[-1])}万人，ARPU为{_fmt(arpu[-1])}元，转化率为{_fmt(conversion[-1])}%。",
        f"图中2025年Active Users、ARPU和Conversion Rate对应数值分别为{_fmt(users[-1])}、{_fmt(arpu[-1])}和{_fmt(conversion[-1])}。",
    )
    records.append(
        _record(
            company,
            "operations",
            image_path,
            "current_value",
            "value_retrieval",
            "easy",
            "读取2025年的活跃用户、ARPU和转化率，并注明单位。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("users_2025", users[-1], "万人"),
                    _numeric("arpu_2025", arpu[-1], "元"),
                    _numeric("conversion_2025", conversion[-1], "%", tolerance=0.2),
                ],
                evidence_keywords=["用户", "ARPU", "转化率"],
                max_chars=240,
            ),
        )
    )

    conclusion = (
        "用户扩张快于收入增长，单位用户变现能力需要改善"
        if facts["user_change_pct"] > facts["revenue_change_pct"] + 10.0
        else "收入增长快于用户扩张，单位用户贡献有所增强"
        if facts["revenue_change_pct"] > facts["user_change_pct"] + 10.0
        else "用户与收入基本同步变化，变现效率总体稳定"
    )
    answer, sections = _answer(
        conclusion + "。",
        f"{_trend_phrase('用户', facts['user_trend'])}，累计变化{_fmt(facts['user_change_pct'])}%；"
        f"收入累计变化{_fmt(facts['revenue_change_pct'])}%；"
        f"{_trend_phrase('ARPU', facts['arpu_trend'])}，累计变化{_fmt(facts['arpu_change_pct'])}%。",
    )
    records.append(
        _record(
            company,
            "operations",
            image_path,
            "monetization",
            "relationship_analysis",
            "medium",
            "比较用户规模与收入的变化，并判断ARPU所反映的变现效率。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("user_change_pct", facts["user_change_pct"], "%", tolerance=0.3),
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("arpu_change_pct", facts["arpu_change_pct"], "%", tolerance=0.3),
                ],
                trend_targets=[
                    _trend("用户", facts["user_trend"]),
                    _trend("ARPU", facts["arpu_trend"]),
                ],
                evidence_keywords=["用户", "收入", "ARPU"],
            ),
        )
    )

    cac_change = percentage_change(cac[0], cac[-1])
    answer, sections = _answer(
        "运营效率需要结合获客成本和转化率共同判断。",
        f"转化率由{_fmt(conversion[0])}%变为{_fmt(conversion[-1])}%，CAC由{_fmt(cac[0])}元变为{_fmt(cac[-1])}元，累计变化{_fmt(cac_change)}%。",
        risk=_risk_sentence(company),
    )
    records.append(
        _record(
            company,
            "operations",
            image_path,
            "efficiency_risk",
            "risk_diagnosis",
            "hard",
            "结合转化率和获客成本判断运营效率风险，并引用首尾数据。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("conversion_2021", conversion[0], "%", tolerance=0.2),
                    _numeric("conversion_2025", conversion[-1], "%", tolerance=0.2),
                    _numeric("cac_change_pct", cac_change, "%", tolerance=0.3),
                ],
                evidence_keywords=["转化率", "CAC", "获客成本"],
                risk_labels=company["facts"]["risks"][:2],
            ),
        )
    )

    answer, sections = _answer(
        "运营优化应同时改善留存、付费转化与获客投入回报。",
        f"2025年用户{_fmt(users[-1])}万人、转化率{_fmt(conversion[-1])}%、CAC为{_fmt(cac[-1])}元。",
        risk=_risk_sentence(company),
        recommendation=_recommendation_sentence(company),
    )
    records.append(
        _record(
            company,
            "operations",
            image_path,
            "decision",
            "decision_support",
            "hard",
            "基于运营指标提出两项优先级明确的改进建议，并说明数据依据。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("users_2025", users[-1], "万人"),
                    _numeric("conversion_2025", conversion[-1], "%", tolerance=0.2),
                    _numeric("cac_2025", cac[-1], "元"),
                ],
                evidence_keywords=["用户", "转化率", "CAC"],
                risk_labels=company["facts"]["risks"][:2],
                grpo_eligible=False,
                min_chars=100,
                max_chars=500,
            ),
        )
    )
    return records


def _growth_records(company: dict[str, Any], image_path: str) -> list[dict[str, Any]]:
    facts = company["facts"]
    years = company["years"]
    revenue_yoy = facts["revenue_yoy_pct"]
    user_yoy = facts["user_yoy_pct"]
    valid_pairs = list(zip(years[1:], revenue_yoy[1:]))
    max_year, max_value = max(valid_pairs, key=lambda pair: pair[1])
    latest_revenue_yoy, latest_user_yoy = revenue_yoy[-1], user_yoy[-1]
    records: list[dict[str, Any]] = []

    answer, sections = _answer(
        f"收入同比增速最高的是{max_year}年，为{_fmt(max_value)}%。",
        "比较2022—2025年Revenue YoY曲线的四个点，最高点对应上述年份和数值。",
    )
    records.append(
        _record(
            company,
            "growth",
            image_path,
            "peak_growth",
            "value_retrieval",
            "easy",
            "哪一年的收入同比增速最高？给出年份和增速。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _year_target("peak_revenue_yoy_year", max_year, terms=["最高", "收入同比"]),
                    _numeric("peak_revenue_yoy", max_value, "%", tolerance=0.2, terms=["最高", "收入同比"]),
                ],
                evidence_keywords=["收入", "同比"],
                max_chars=220,
            ),
        )
    )

    gap = latest_revenue_yoy - latest_user_yoy
    conclusion = (
        "2025年收入增速高于用户增速，单用户贡献倾向改善"
        if gap > 2.0
        else "2025年用户增速高于收入增速，变现效率可能承压"
        if gap < -2.0
        else "2025年收入与用户增速接近，二者基本同步"
    )
    answer, sections = _answer(
        conclusion + "。",
        f"2025年Revenue YoY为{_fmt(latest_revenue_yoy)}%，User YoY为{_fmt(latest_user_yoy)}%，增速差为{_fmt(gap)}个百分点。",
    )
    records.append(
        _record(
            company,
            "growth",
            image_path,
            "growth_gap",
            "numerical_reasoning",
            "medium",
            "比较2025年收入同比与用户同比，计算增速差并解释含义。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_yoy_2025", latest_revenue_yoy, "%", tolerance=0.2),
                    _numeric("user_yoy_2025", latest_user_yoy, "%", tolerance=0.2),
                    _numeric("growth_gap_pp", gap, "个百分点", tolerance=0.3, terms=["增速差"]),
                ],
                evidence_keywords=["收入", "用户", "增速差"],
            ),
        )
    )

    answer, sections = _answer(
        f"{_trend_phrase('收入', facts['revenue_trend'])}；{_trend_phrase('用户', facts['user_trend'])}。增长动能还需结合最新同比判断。",
        f"收入累计变化{_fmt(facts['revenue_change_pct'])}%，2025年同比{_fmt(latest_revenue_yoy)}%；"
        f"用户累计变化{_fmt(facts['user_change_pct'])}%，2025年同比{_fmt(latest_user_yoy)}%。",
        risk=_risk_sentence(company, 1),
    )
    records.append(
        _record(
            company,
            "growth",
            image_path,
            "trend",
            "trend_analysis",
            "hard",
            "判断收入与用户增长动能是增强、减弱还是分化，并用累计与最新同比数据说明。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("user_change_pct", facts["user_change_pct"], "%", tolerance=0.3),
                    _numeric("revenue_yoy_2025", latest_revenue_yoy, "%", tolerance=0.2),
                    _numeric("user_yoy_2025", latest_user_yoy, "%", tolerance=0.2),
                ],
                trend_targets=[_trend("收入", facts["revenue_trend"]), _trend("用户", facts["user_trend"])],
                evidence_keywords=["收入", "用户", "同比"],
                risk_labels=company["facts"]["risks"][:1],
            ),
        )
    )

    answer, sections = _answer(
        "建议将增长目标从规模扩张升级为规模与效率的联合目标。",
        f"2025年收入同比{_fmt(latest_revenue_yoy)}%，用户同比{_fmt(latest_user_yoy)}%，收入CAGR为{_fmt(facts['revenue_cagr_pct'])}%。",
        risk=_risk_sentence(company),
        recommendation=_recommendation_sentence(company),
    )
    records.append(
        _record(
            company,
            "growth",
            image_path,
            "decision",
            "decision_support",
            "hard",
            "根据增长率图表提出下一年度经营重点，并说明主要风险。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_yoy_2025", latest_revenue_yoy, "%", tolerance=0.2),
                    _numeric("user_yoy_2025", latest_user_yoy, "%", tolerance=0.2),
                    _numeric("revenue_cagr_pct", facts["revenue_cagr_pct"], "%", tolerance=0.3),
                ],
                evidence_keywords=["收入", "用户", "CAGR"],
                risk_labels=company["facts"]["risks"][:2],
                grpo_eligible=False,
                min_chars=100,
                max_chars=500,
            ),
        )
    )
    return records


def _composition_records(company: dict[str, Any], image_path: str) -> list[dict[str, Any]]:
    product_mix = company["product_mix_pct"]
    region_mix = company["region_mix_pct"]
    product_sorted = sorted(product_mix.items(), key=lambda pair: pair[1], reverse=True)
    region_sorted = sorted(region_mix.items(), key=lambda pair: pair[1], reverse=True)
    top_product, top_product_share = product_sorted[0]
    top_region, top_region_share = region_sorted[0]
    top_two_share = product_sorted[0][1] + product_sorted[1][1]
    concentrated = top_product_share >= 55.0
    records: list[dict[str, Any]] = []

    answer, sections = _answer(
        f"占比最高的产品是{top_product}，收入占比为{_fmt(top_product_share)}%。",
        f"产品结构饼图中{top_product}扇区最大；第二大产品为{product_sorted[1][0]}，占比{_fmt(product_sorted[1][1])}%。",
    )
    records.append(
        _record(
            company,
            "composition",
            image_path,
            "top_product",
            "value_retrieval",
            "easy",
            "收入占比最高的产品是什么？占比多少？",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("top_product_share", top_product_share, "%", tolerance=0.2, terms=[top_product])
                ],
                evidence_keywords=[top_product, "占比"],
                max_chars=220,
            ),
        )
    )

    answer, sections = _answer(
        f"前两大产品合计占收入{_fmt(top_two_share)}%。",
        f"{product_sorted[0][0]}占{_fmt(product_sorted[0][1])}%，{product_sorted[1][0]}占{_fmt(product_sorted[1][1])}%，两者相加得到{_fmt(top_two_share)}%。",
    )
    records.append(
        _record(
            company,
            "composition",
            image_path,
            "top_two",
            "numerical_reasoning",
            "medium",
            "计算前两大产品的合计收入占比，并列出计算依据。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric(
                        "product_1_share",
                        product_sorted[0][1],
                        "%",
                        tolerance=0.2,
                        terms=[product_sorted[0][0]],
                    ),
                    _numeric(
                        "product_2_share",
                        product_sorted[1][1],
                        "%",
                        tolerance=0.2,
                        terms=[product_sorted[1][0]],
                    ),
                    _numeric("top_two_share", top_two_share, "%", tolerance=0.3, terms=["前两大", "合计"]),
                ],
                evidence_keywords=[product_sorted[0][0], product_sorted[1][0]],
                max_chars=280,
            ),
        )
    )

    concentration_text = "产品结构集中度偏高" if concentrated else "单一产品依赖尚不突出"
    answer, sections = _answer(
        f"{concentration_text}，同时最大区域为{top_region}。",
        f"第一大产品{top_product}占{_fmt(top_product_share)}%，第一大区域{top_region}占{_fmt(top_region_share)}%。",
        risk=("核心产品波动会对整体收入造成较大冲击。" if concentrated else "仍需监控前两大产品和区域的同步波动。"),
    )
    records.append(
        _record(
            company,
            "composition",
            image_path,
            "concentration",
            "risk_diagnosis",
            "medium",
            "判断产品与区域结构是否存在集中度风险，并引用最大占比。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric(
                        "top_product_share", top_product_share, "%", tolerance=0.2, terms=[top_product]
                    ),
                    _numeric("top_region_share", top_region_share, "%", tolerance=0.2, terms=[top_region]),
                ],
                evidence_keywords=["产品", "区域", top_product, top_region],
                risk_labels=["product_concentration"] if concentrated else [],
            ),
        )
    )

    answer, sections = _answer(
        "结构优化应兼顾巩固核心收入与培育增量来源。",
        f"{top_product}占{_fmt(top_product_share)}%，{top_region}占{_fmt(top_region_share)}%，前两大产品合计{_fmt(top_two_share)}%。",
        risk=("产品收入集中度偏高。" if concentrated else "结构相对分散，但仍存在组合波动风险。"),
        recommendation=(
            "设定非核心产品增长目标，并在非优势区域验证新的获客渠道。"
            if concentrated
            else "保持多元结构，同时依据利润贡献而非收入占比配置资源。"
        ),
    )
    records.append(
        _record(
            company,
            "composition",
            image_path,
            "decision",
            "decision_support",
            "hard",
            "基于产品和区域收入结构提出资源配置建议。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric(
                        "top_product_share", top_product_share, "%", tolerance=0.2, terms=[top_product]
                    ),
                    _numeric("top_region_share", top_region_share, "%", tolerance=0.2, terms=[top_region]),
                    _numeric("top_two_share", top_two_share, "%", tolerance=0.3, terms=["前两大", "合计"]),
                ],
                evidence_keywords=[top_product, top_region, "产品"],
                risk_labels=["product_concentration"] if concentrated else [],
                grpo_eligible=False,
                min_chars=95,
                max_chars=480,
            ),
        )
    )
    return records


def _dashboard_records(company: dict[str, Any], image_path: str) -> list[dict[str, Any]]:
    metrics, facts = company["metrics"], company["facts"]
    revenue, profit = metrics["revenue_million"], metrics["profit_million"]
    users, margin = metrics["active_users_10k"], metrics["profit_margin_pct"]
    conversion = metrics["conversion_rate_pct"]
    records: list[dict[str, Any]] = []

    answer, sections = _answer(
        f"2025年收入{_fmt(revenue[-1])}百万元、利润{_fmt(profit[-1])}百万元、活跃用户{_fmt(users[-1])}万人、利润率{_fmt(margin[-1])}%。",
        "四项数值分别来自仪表盘的财务、利润率和用户面板。",
    )
    records.append(
        _record(
            company,
            "dashboard",
            image_path,
            "snapshot",
            "value_retrieval",
            "medium",
            "概括2025年的收入、利润、活跃用户和利润率四项核心指标。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_2025", revenue[-1], "百万元"),
                    _numeric("profit_2025", profit[-1], "百万元"),
                    _numeric("users_2025", users[-1], "万人"),
                    _numeric("margin_2025", margin[-1], "%", tolerance=0.2),
                ],
                evidence_keywords=["收入", "利润", "用户", "利润率"],
                max_chars=260,
            ),
        )
    )

    answer, sections = _answer(
        "规模、变现和盈利需要联合判断，单看收入会遗漏增长质量。",
        f"{_trend_phrase('收入', facts['revenue_trend'])}，累计变化{_fmt(facts['revenue_change_pct'])}%；"
        f"{_trend_phrase('用户', facts['user_trend'])}，累计变化{_fmt(facts['user_change_pct'])}%；"
        f"ARPU变化{_fmt(facts['arpu_change_pct'])}%；"
        f"{_trend_phrase('利润率', facts['margin_trend'])}，变化{_fmt(facts['margin_change_pp'])}个百分点。",
    )
    records.append(
        _record(
            company,
            "dashboard",
            image_path,
            "multi_metric",
            "relationship_analysis",
            "hard",
            "从规模、用户变现和盈利质量三个层面分析经营变化。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("user_change_pct", facts["user_change_pct"], "%", tolerance=0.3),
                    _numeric("arpu_change_pct", facts["arpu_change_pct"], "%", tolerance=0.3),
                    _numeric("margin_change_pp", facts["margin_change_pp"], "个百分点", tolerance=0.3),
                ],
                trend_targets=[
                    _trend("收入", facts["revenue_trend"]),
                    _trend("用户", facts["user_trend"]),
                    _trend("利润率", facts["margin_trend"]),
                ],
                evidence_keywords=["收入", "用户", "ARPU", "利润率"],
            ),
        )
    )

    answer, sections = _answer(
        "当前经营风险来自规模与效率指标的组合变化。",
        f"收入变化{_fmt(facts['revenue_change_pct'])}%，用户变化{_fmt(facts['user_change_pct'])}%，利润率变化{_fmt(facts['margin_change_pp'])}个百分点，2025年转化率{_fmt(conversion[-1])}%。",
        risk=_risk_sentence(company),
    )
    records.append(
        _record(
            company,
            "dashboard",
            image_path,
            "risk",
            "risk_diagnosis",
            "hard",
            "识别仪表盘中最重要的两项经营风险，并给出量化依据。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("user_change_pct", facts["user_change_pct"], "%", tolerance=0.3),
                    _numeric("margin_change_pp", facts["margin_change_pp"], "个百分点", tolerance=0.3),
                ],
                evidence_keywords=["收入", "用户", "利润率"],
                risk_labels=company["facts"]["risks"][:2],
            ),
        )
    )

    answer, sections = _answer(
        "经营策略应围绕增长质量设定联合指标，而不是追求单一规模指标。",
        f"收入变化{_fmt(facts['revenue_change_pct'])}%，用户变化{_fmt(facts['user_change_pct'])}%，利润率变化{_fmt(facts['margin_change_pp'])}个百分点。",
        risk=_risk_sentence(company),
        recommendation=_recommendation_sentence(company),
    )
    records.append(
        _record(
            company,
            "dashboard",
            image_path,
            "report",
            "comprehensive_report",
            "hard",
            "生成一份简洁的经营分析报告，包含结论、数据依据、风险和建议。",
            answer,
            _base_ground_truth(
                sections,
                numeric_targets=[
                    _numeric("revenue_change_pct", facts["revenue_change_pct"], "%", tolerance=0.3),
                    _numeric("user_change_pct", facts["user_change_pct"], "%", tolerance=0.3),
                    _numeric("margin_change_pp", facts["margin_change_pp"], "个百分点", tolerance=0.3),
                ],
                evidence_keywords=["收入", "用户", "利润率"],
                risk_labels=company["facts"]["risks"][:2],
                grpo_eligible=False,
                min_chars=110,
                max_chars=560,
            ),
        )
    )
    return records


BUILDERS = {
    "financial": _financial_records,
    "operations": _operations_records,
    "growth": _growth_records,
    "composition": _composition_records,
    "dashboard": _dashboard_records,
}


def build_records_for_chart(
    company: dict[str, Any], view_type: str, image_path: str, seed: int
) -> list[dict[str, Any]]:
    records = BUILDERS[view_type](company, image_path)
    # Reorder tasks independently of the business scenario so training does not
    # inherit a fixed task sequence from the generator.
    random.Random(stable_seed(f"question_order:{company['entity_id']}:{view_type}", seed)).shuffle(records)
    return records
