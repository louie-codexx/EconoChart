from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from econochart.io import stable_seed

INDUSTRIES: dict[str, dict[str, Any]] = {
    "ecommerce": {
        "label_zh": "电子商务",
        "arpu_range": (150.0, 360.0),
        "products": ["Marketplace", "Direct Sales", "Advertising"],
    },
    "saas": {
        "label_zh": "企业软件服务",
        "arpu_range": (280.0, 620.0),
        "products": ["Subscription", "Implementation", "Add-ons"],
    },
    "fintech": {
        "label_zh": "金融科技",
        "arpu_range": (180.0, 480.0),
        "products": ["Payments", "Credit Tech", "Wealth Tech"],
    },
    "digital_media": {
        "label_zh": "数字内容",
        "arpu_range": (80.0, 260.0),
        "products": ["Subscriptions", "Advertising", "Licensing"],
    },
    "cloud_services": {
        "label_zh": "云服务",
        "arpu_range": (320.0, 760.0),
        "products": ["Compute", "Storage", "Data Services"],
    },
    "local_services": {
        "label_zh": "本地生活服务",
        "arpu_range": (110.0, 300.0),
        "products": ["Transactions", "Marketing", "Membership"],
    },
}

SCENARIO_PATTERNS: dict[str, dict[str, list[float]]] = {
    "high_growth": {
        "user_growth": [0.18, 0.24, 0.22, 0.19],
        "arpu_growth": [0.04, 0.06, 0.05, 0.04],
        "cost_ratio_delta": [-0.008, -0.010, -0.012, -0.010],
        "conversion_delta": [0.35, 0.45, 0.40, 0.30],
        "cac_growth": [0.04, 0.03, 0.02, 0.02],
    },
    "stable": {
        "user_growth": [0.04, 0.03, 0.05, 0.03],
        "arpu_growth": [0.01, 0.02, 0.01, 0.02],
        "cost_ratio_delta": [0.002, -0.002, 0.001, -0.001],
        "conversion_delta": [0.05, 0.02, 0.08, 0.04],
        "cac_growth": [0.01, 0.02, 0.01, 0.02],
    },
    "cost_pressure": {
        "user_growth": [0.10, 0.12, 0.09, 0.08],
        "arpu_growth": [0.03, 0.03, 0.02, 0.02],
        "cost_ratio_delta": [0.035, 0.045, 0.050, 0.040],
        "conversion_delta": [0.10, 0.05, -0.05, -0.10],
        "cac_growth": [0.12, 0.15, 0.18, 0.14],
    },
    "demand_slowdown": {
        "user_growth": [0.15, 0.08, 0.01, -0.07],
        "arpu_growth": [0.04, 0.02, 0.00, -0.03],
        "cost_ratio_delta": [-0.005, 0.005, 0.018, 0.025],
        "conversion_delta": [0.20, 0.05, -0.20, -0.35],
        "cac_growth": [0.04, 0.08, 0.12, 0.15],
    },
    "user_churn": {
        "user_growth": [-0.06, -0.10, -0.13, -0.09],
        "arpu_growth": [0.02, 0.03, 0.01, -0.01],
        "cost_ratio_delta": [0.010, 0.018, 0.025, 0.020],
        "conversion_delta": [-0.25, -0.35, -0.45, -0.30],
        "cac_growth": [0.08, 0.12, 0.15, 0.10],
    },
    "recovery": {
        "user_growth": [-0.09, -0.03, 0.09, 0.17],
        "arpu_growth": [-0.03, 0.00, 0.04, 0.06],
        "cost_ratio_delta": [0.025, 0.010, -0.025, -0.035],
        "conversion_delta": [-0.30, -0.10, 0.30, 0.50],
        "cac_growth": [0.10, 0.04, -0.06, -0.08],
    },
    "margin_expansion": {
        "user_growth": [0.07, 0.08, 0.09, 0.08],
        "arpu_growth": [0.03, 0.04, 0.04, 0.03],
        "cost_ratio_delta": [-0.025, -0.030, -0.025, -0.020],
        "conversion_delta": [0.15, 0.20, 0.20, 0.15],
        "cac_growth": [-0.02, -0.03, -0.03, -0.02],
    },
    "monetization_weakness": {
        "user_growth": [0.20, 0.18, 0.16, 0.14],
        "arpu_growth": [-0.12, -0.10, -0.08, -0.06],
        "cost_ratio_delta": [0.010, 0.015, 0.020, 0.015],
        "conversion_delta": [0.30, 0.25, 0.15, 0.10],
        "cac_growth": [0.06, 0.08, 0.09, 0.08],
    },
}

SECONDARY_VIEWS = ("operations", "growth", "composition", "dashboard")
REGIONS = ("East", "South", "North", "West")


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _round_series(values: list[float], digits: int = 1) -> list[float]:
    return [round(value, digits) for value in values]


def percentage_change(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def cagr(start: float, end: float, periods: int) -> float:
    if start <= 0 or end <= 0 or periods <= 0:
        return 0.0
    return ((end / start) ** (1.0 / periods) - 1.0) * 100.0


def _shares(labels: list[str] | tuple[str, ...], rng: random.Random, concentrated: bool) -> dict[str, float]:
    if concentrated:
        dominant_index = rng.randrange(len(labels))
        dominant = rng.uniform(0.56, 0.72)
        residual_weights = [rng.uniform(0.4, 1.4) for _ in labels]
        residual_weights[dominant_index] = 0.0
        residual_total = sum(residual_weights)
        raw = [
            dominant if index == dominant_index else (1.0 - dominant) * residual_weights[index] / residual_total
            for index in range(len(labels))
        ]
    else:
        weights = [rng.gammavariate(3.0, 1.0) for _ in labels]
        total = sum(weights)
        raw = [weight / total for weight in weights]
    rounded = [round(value * 100.0, 1) for value in raw]
    rounded[-1] = round(100.0 - sum(rounded[:-1]), 1)
    return dict(zip(labels, rounded))


def _trend_direction(values: list[float], stable_threshold: float = 3.0) -> str:
    change = percentage_change(values[0], values[-1])
    if change > stable_threshold:
        return "increase"
    if change < -stable_threshold:
        return "decrease"
    return "stable"


def derive_facts(company: dict[str, Any]) -> dict[str, Any]:
    revenue = company["metrics"]["revenue_million"]
    cost = company["metrics"]["cost_million"]
    profit = company["metrics"]["profit_million"]
    users = company["metrics"]["active_users_10k"]
    arpu = company["metrics"]["arpu_yuan"]
    margin = company["metrics"]["profit_margin_pct"]
    conversion = company["metrics"]["conversion_rate_pct"]

    revenue_growth = percentage_change(revenue[0], revenue[-1])
    cost_growth = percentage_change(cost[0], cost[-1])
    user_growth = percentage_change(users[0], users[-1])
    arpu_growth = percentage_change(arpu[0], arpu[-1])
    margin_change = margin[-1] - margin[0]
    revenue_yoy = [None] + [round(percentage_change(revenue[index - 1], revenue[index]), 1) for index in range(1, 5)]
    user_yoy = [None] + [round(percentage_change(users[index - 1], users[index]), 1) for index in range(1, 5)]

    risks: list[str] = []
    if revenue_growth < -5.0:
        risks.append("revenue_decline")
    if user_growth < -8.0:
        risks.append("user_churn")
    if margin_change < -5.0:
        risks.append("margin_pressure")
    if any(value < 0 for value in profit):
        risks.append("loss_making")
    if user_growth > 20.0 and revenue_growth < user_growth - 10.0:
        risks.append("monetization_weakness")
    if company["metrics"]["cac_yuan"][-1] > company["metrics"]["cac_yuan"][0] * 1.35:
        risks.append("acquisition_cost_pressure")
    if max(company["product_mix_pct"].values()) >= 55.0:
        risks.append("product_concentration")
    if not risks:
        risks.append("execution_volatility")

    strengths: list[str] = []
    if revenue_growth > 25.0:
        strengths.append("revenue_growth")
    if user_growth > 20.0:
        strengths.append("user_growth")
    if margin_change > 4.0:
        strengths.append("margin_expansion")
    if conversion[-1] > conversion[0] + 0.8:
        strengths.append("conversion_improvement")

    return {
        "revenue_change_pct": round(revenue_growth, 1),
        "cost_change_pct": round(cost_growth, 1),
        "user_change_pct": round(user_growth, 1),
        "arpu_change_pct": round(arpu_growth, 1),
        "margin_change_pp": round(margin_change, 1),
        "revenue_cagr_pct": round(cagr(revenue[0], revenue[-1], 4), 1),
        "revenue_yoy_pct": revenue_yoy,
        "user_yoy_pct": user_yoy,
        "revenue_trend": _trend_direction(revenue),
        "cost_trend": _trend_direction(cost),
        "profit_trend": _trend_direction(profit, stable_threshold=5.0),
        "user_trend": _trend_direction(users),
        "arpu_trend": _trend_direction(arpu),
        "margin_trend": "increase" if margin_change > 2.0 else "decrease" if margin_change < -2.0 else "stable",
        "risks": risks,
        "strengths": strengths,
    }


def generate_company(index: int, industry: str, scenario: str, seed: int) -> dict[str, Any]:
    entity_id = f"ec_company_{index:06d}"
    rng = random.Random(stable_seed(f"business:{entity_id}", seed))
    mix_rng = random.Random(stable_seed(f"mix:{entity_id}", seed))
    pattern = SCENARIO_PATTERNS[scenario]
    industry_config = INDUSTRIES[industry]
    years = [2021, 2022, 2023, 2024, 2025]

    users = [rng.uniform(55.0, 420.0)]
    arpu = [rng.uniform(*industry_config["arpu_range"])]
    cost_ratio = [rng.uniform(0.62, 0.88)]
    conversion = [rng.uniform(2.5, 11.0)]
    cac = [rng.uniform(35.0, 180.0)]

    for step in range(4):
        user_growth = pattern["user_growth"][step] + rng.uniform(-0.018, 0.018)
        arpu_growth = pattern["arpu_growth"][step] + rng.uniform(-0.012, 0.012)
        users.append(max(8.0, users[-1] * (1.0 + user_growth)))
        arpu.append(max(25.0, arpu[-1] * (1.0 + arpu_growth)))
        cost_ratio.append(
            _bounded(cost_ratio[-1] + pattern["cost_ratio_delta"][step] + rng.uniform(-0.006, 0.006), 0.45, 1.14)
        )
        conversion.append(
            _bounded(conversion[-1] + pattern["conversion_delta"][step] + rng.uniform(-0.08, 0.08), 0.8, 22.0)
        )
        cac.append(
            max(10.0, cac[-1] * (1.0 + pattern["cac_growth"][step] + rng.uniform(-0.015, 0.015)))
        )

    revenue = [user_value * arpu_value / 100.0 for user_value, arpu_value in zip(users, arpu)]
    cost = [revenue_value * ratio for revenue_value, ratio in zip(revenue, cost_ratio)]
    profit = [revenue_value - cost_value for revenue_value, cost_value in zip(revenue, cost)]
    margin = [100.0 * profit_value / revenue_value for profit_value, revenue_value in zip(profit, revenue)]

    product_mix = _shares(industry_config["products"], mix_rng, concentrated=mix_rng.random() < 0.28)
    region_mix = _shares(REGIONS, mix_rng, concentrated=mix_rng.random() < 0.18)
    secondary_view = SECONDARY_VIEWS[stable_seed(f"view:{entity_id}", seed) % len(SECONDARY_VIEWS)]

    company = {
        "entity_id": entity_id,
        "industry": industry,
        "industry_zh": industry_config["label_zh"],
        "scenario": scenario,
        "years": years,
        "metrics": {
            "revenue_million": _round_series(revenue),
            "cost_million": _round_series(cost),
            "profit_million": _round_series(profit),
            "profit_margin_pct": _round_series(margin),
            "active_users_10k": _round_series(users),
            "arpu_yuan": _round_series(arpu),
            "conversion_rate_pct": _round_series(conversion),
            "cac_yuan": _round_series(cac),
        },
        "product_mix_pct": product_mix,
        "region_mix_pct": region_mix,
        "views": ["financial", secondary_view],
    }
    company["facts"] = derive_facts(company)
    return company


def _assign_global_splits(
    companies: list[dict[str, Any]], split_ratios: dict[str, float], seed: int
) -> None:
    shuffled = list(companies)
    random.Random(stable_seed("global_split", seed)).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * split_ratios["train"])
    val_end = train_end + int(total * split_ratios["val"])
    if total >= 3:
        train_end = min(max(train_end, 1), total - 2)
        val_end = min(max(val_end, train_end + 1), total - 1)
    for position, company in enumerate(shuffled):
        company["split"] = "train" if position < train_end else "val" if position < val_end else "test"


def assign_splits(companies: list[dict[str, Any]], split_ratios: dict[str, float], seed: int) -> None:
    if set(split_ratios) != {"train", "val", "test"}:
        raise ValueError("split_ratios must contain exactly train, val, and test")
    if not math.isclose(sum(split_ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1.0")
    if any(value <= 0 for value in split_ratios.values()):
        raise ValueError("all split ratios must be positive")

    # Small sample builds are for repository smoke tests, where global splitting
    # guarantees every split is represented. Full builds are stratified.
    if len(companies) < 100:
        _assign_global_splits(companies, split_ratios, seed)
        return

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for company in companies:
        buckets[(company["industry"], company["scenario"])].append(company)

    for (industry, scenario), bucket in sorted(buckets.items()):
        random.Random(stable_seed(f"split:{industry}:{scenario}", seed)).shuffle(bucket)
        total = len(bucket)
        train_count = int(round(total * split_ratios["train"]))
        val_count = int(round(total * split_ratios["val"]))
        train_count = min(max(train_count, 1), total - 2)
        val_count = min(max(val_count, 1), total - train_count - 1)
        for position, company in enumerate(bucket):
            if position < train_count:
                company["split"] = "train"
            elif position < train_count + val_count:
                company["split"] = "val"
            else:
                company["split"] = "test"


def generate_companies(
    num_companies: int,
    seed: int,
    split_ratios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if num_companies < 3:
        raise ValueError("num_companies must be at least 3")
    ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    combinations = [
        (industry, scenario)
        for industry in sorted(INDUSTRIES)
        for scenario in sorted(SCENARIO_PATTERNS)
    ]
    combo_rng = random.Random(stable_seed("balanced_combinations", seed))
    combo_rng.shuffle(combinations)
    companies = [
        generate_company(
            index=index,
            industry=combinations[(index - 1) % len(combinations)][0],
            scenario=combinations[(index - 1) % len(combinations)][1],
            seed=seed,
        )
        for index in range(1, num_companies + 1)
    ]
    assign_splits(companies, ratios, seed)
    return companies
