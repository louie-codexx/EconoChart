from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from econochart.io import stable_seed

STYLE_PRESETS = (
    {
        "name": "blue_orange",
        "colors": ("#2F6BFF", "#F59E0B", "#10B981", "#EF4444"),
        "background": "#FFFFFF",
        "grid_alpha": 0.22,
        "marker": "o",
    },
    {
        "name": "teal_purple",
        "colors": ("#0F766E", "#7C3AED", "#E11D48", "#0284C7"),
        "background": "#FAFAFA",
        "grid_alpha": 0.18,
        "marker": "s",
    },
    {
        "name": "navy_coral",
        "colors": ("#1E3A8A", "#FB7185", "#22C55E", "#A855F7"),
        "background": "#F8FAFC",
        "grid_alpha": 0.20,
        "marker": "D",
    },
    {
        "name": "charcoal_cyan",
        "colors": ("#334155", "#06B6D4", "#F97316", "#84CC16"),
        "background": "#FFFFFF",
        "grid_alpha": 0.16,
        "marker": "^",
    },
)


def _style(company_id: str, view_type: str, seed: int, label_probability: float) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"chart_style:{company_id}:{view_type}", seed))
    preset = dict(STYLE_PRESETS[rng.randrange(len(STYLE_PRESETS))])
    preset["show_labels"] = rng.random() < label_probability
    preset["line_width"] = rng.choice((2.0, 2.3, 2.6))
    preset["legend_location"] = rng.choice(("best", "upper left", "upper right"))
    return preset


def _annotate_points(axis: Any, x_values: list[Any], y_values: list[float], color: str) -> None:
    for x_value, y_value in zip(x_values, y_values):
        axis.annotate(
            f"{y_value:.1f}",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
        )


def _style_axis(axis: Any, background: str, grid_alpha: float) -> None:
    axis.set_facecolor(background)
    axis.grid(axis="y", alpha=grid_alpha, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_alpha(0.25)


def _render_financial(fig: Any, company: dict[str, Any], style: dict[str, Any], np: Any) -> None:
    axis = fig.add_subplot(111)
    years = company["years"]
    metrics = company["metrics"]
    x_values = np.arange(len(years))
    width = 0.29
    revenue = metrics["revenue_million"]
    cost = metrics["cost_million"]
    profit = metrics["profit_million"]
    colors = style["colors"]

    revenue_bars = axis.bar(x_values - width / 2, revenue, width, label="Revenue", color=colors[0], alpha=0.88)
    cost_bars = axis.bar(x_values + width / 2, cost, width, label="Cost", color=colors[1], alpha=0.82)
    axis.plot(
        x_values,
        profit,
        label="Profit",
        color=colors[2],
        marker=style["marker"],
        linewidth=style["line_width"],
        zorder=4,
    )
    axis.axhline(0, color="#64748B", linewidth=0.9)
    axis.set_xticks(x_values, years)
    axis.set_xlabel("Year")
    axis.set_ylabel("RMB million")
    axis.legend(loc=style["legend_location"], frameon=False, ncols=3)
    _style_axis(axis, style["background"], style["grid_alpha"])
    if style["show_labels"]:
        axis.bar_label(revenue_bars, fmt="%.1f", fontsize=7, padding=2)
        axis.bar_label(cost_bars, fmt="%.1f", fontsize=7, padding=2)
        _annotate_points(axis, list(x_values), profit, colors[2])


def _combine_legends(primary: Any, secondary: Any, location: str) -> None:
    handles_a, labels_a = primary.get_legend_handles_labels()
    handles_b, labels_b = secondary.get_legend_handles_labels()
    primary.legend(handles_a + handles_b, labels_a + labels_b, loc=location, frameon=False, fontsize=8)


def _render_operations(fig: Any, company: dict[str, Any], style: dict[str, Any], np: Any) -> None:
    years = company["years"]
    metrics = company["metrics"]
    colors = style["colors"]
    upper = fig.add_subplot(211)
    upper_right = upper.twinx()
    lower = fig.add_subplot(212)
    lower_right = lower.twinx()

    upper.plot(years, metrics["active_users_10k"], color=colors[0], marker=style["marker"], label="Active Users")
    upper_right.plot(years, metrics["arpu_yuan"], color=colors[1], marker="o", linestyle="--", label="ARPU")
    upper.set_ylabel("Users (10k)", color=colors[0])
    upper_right.set_ylabel("ARPU (RMB)", color=colors[1])
    upper.set_xticks(years)
    upper.tick_params(axis="x", labelbottom=False)
    _combine_legends(upper, upper_right, style["legend_location"])

    lower.plot(
        years,
        metrics["conversion_rate_pct"],
        color=colors[2],
        marker=style["marker"],
        label="Conversion Rate",
    )
    lower_right.plot(years, metrics["cac_yuan"], color=colors[3], marker="o", linestyle="--", label="CAC")
    lower.set_ylabel("Conversion (%)", color=colors[2])
    lower_right.set_ylabel("CAC (RMB)", color=colors[3])
    lower.set_xlabel("Year")
    lower.set_xticks(years)
    _combine_legends(lower, lower_right, style["legend_location"])

    for axis in (upper, lower):
        _style_axis(axis, style["background"], style["grid_alpha"])
    for axis in (upper_right, lower_right):
        axis.spines["top"].set_visible(False)
    if style["show_labels"]:
        _annotate_points(upper, years, metrics["active_users_10k"], colors[0])
        _annotate_points(upper_right, years, metrics["arpu_yuan"], colors[1])
        _annotate_points(lower, years, metrics["conversion_rate_pct"], colors[2])
        _annotate_points(lower_right, years, metrics["cac_yuan"], colors[3])


def _render_growth(fig: Any, company: dict[str, Any], style: dict[str, Any], np: Any) -> None:
    del np
    years = company["years"]
    facts = company["facts"]
    margin = company["metrics"]["profit_margin_pct"]
    colors = style["colors"]
    left = fig.add_subplot(121)
    right = fig.add_subplot(122)
    yoy_years = years[1:]
    revenue_yoy = facts["revenue_yoy_pct"][1:]
    user_yoy = facts["user_yoy_pct"][1:]

    left.plot(yoy_years, revenue_yoy, label="Revenue YoY", color=colors[0], marker=style["marker"], linewidth=2.2)
    left.plot(yoy_years, user_yoy, label="User YoY", color=colors[1], marker="o", linewidth=2.2)
    left.axhline(0, color="#64748B", linewidth=0.9)
    left.set_title("Growth Rates")
    left.set_xlabel("Year")
    left.set_ylabel("YoY (%)")
    left.set_xticks(yoy_years)
    left.legend(frameon=False, fontsize=8)

    right.plot(years, margin, label="Profit Margin", color=colors[2], marker=style["marker"], linewidth=2.2)
    right.axhline(0, color="#64748B", linewidth=0.9)
    right.set_title("Profitability")
    right.set_xlabel("Year")
    right.set_ylabel("Margin (%)")
    right.set_xticks(years)
    right.legend(frameon=False, fontsize=8)

    for axis in (left, right):
        _style_axis(axis, style["background"], style["grid_alpha"])
    if style["show_labels"]:
        _annotate_points(left, yoy_years, revenue_yoy, colors[0])
        _annotate_points(left, yoy_years, user_yoy, colors[1])
        _annotate_points(right, years, margin, colors[2])


def _render_composition(fig: Any, company: dict[str, Any], style: dict[str, Any], np: Any) -> None:
    del np
    left = fig.add_subplot(121)
    right = fig.add_subplot(122)
    colors = style["colors"]
    product_labels = list(company["product_mix_pct"])
    product_values = list(company["product_mix_pct"].values())
    region_labels = list(company["region_mix_pct"])
    region_values = list(company["region_mix_pct"].values())
    autopct = "%1.1f%%" if style["show_labels"] else None

    left.pie(
        product_values,
        labels=product_labels,
        autopct=autopct,
        colors=colors[: len(product_values)],
        startangle=90,
        textprops={"fontsize": 8},
    )
    left.set_title("Revenue by Product")
    right.pie(
        region_values,
        labels=region_labels,
        autopct=autopct,
        colors=colors[: len(region_values)],
        startangle=90,
        textprops={"fontsize": 8},
    )
    right.set_title("Revenue by Region")


def _render_dashboard(fig: Any, company: dict[str, Any], style: dict[str, Any], np: Any) -> None:
    years = company["years"]
    metrics = company["metrics"]
    colors = style["colors"]
    axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]

    axes[0].plot(years, metrics["revenue_million"], label="Revenue", color=colors[0], marker=style["marker"])
    axes[0].plot(years, metrics["cost_million"], label="Cost", color=colors[1], marker="o")
    axes[0].set_title("Revenue & Cost")
    axes[0].set_ylabel("RMB million")

    axes[1].bar(years, metrics["profit_million"], color=colors[2], alpha=0.85, label="Profit")
    axes[1].axhline(0, color="#64748B", linewidth=0.8)
    axes[1].set_title("Profit")
    axes[1].set_ylabel("RMB million")

    axes[2].plot(years, metrics["active_users_10k"], color=colors[0], marker=style["marker"], label="Active Users")
    axes[2].set_title("Active Users")
    axes[2].set_ylabel("10k users")

    axes[3].plot(years, metrics["profit_margin_pct"], color=colors[2], marker=style["marker"], label="Profit Margin")
    axes[3].plot(years, metrics["conversion_rate_pct"], color=colors[3], marker="o", label="Conversion")
    axes[3].axhline(0, color="#64748B", linewidth=0.8)
    axes[3].set_title("Margin & Conversion")
    axes[3].set_ylabel("Percent")

    for axis in axes:
        axis.set_xticks(years)
        axis.tick_params(axis="x", labelrotation=0, labelsize=7)
        axis.legend(frameon=False, fontsize=7)
        _style_axis(axis, style["background"], style["grid_alpha"])

    if style["show_labels"]:
        _annotate_points(axes[0], years, metrics["revenue_million"], colors[0])
        _annotate_points(axes[2], years, metrics["active_users_10k"], colors[0])
        _annotate_points(axes[3], years, metrics["profit_margin_pct"], colors[2])


RENDERERS = {
    "financial": _render_financial,
    "operations": _render_operations,
    "growth": _render_growth,
    "composition": _render_composition,
    "dashboard": _render_dashboard,
}


def render_chart(
    company: dict[str, Any],
    view_type: str,
    output_path: str | Path,
    *,
    seed: int,
    width_px: int = 1280,
    height_px: int = 768,
    dpi: int = 160,
    label_probability: float = 0.5,
) -> dict[str, Any]:
    if view_type not in RENDERERS:
        raise ValueError(f"Unsupported view type: {view_type}")
    if not 0.0 <= label_probability <= 1.0:
        raise ValueError("label_probability must be between 0 and 1")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    style = _style(company["entity_id"], view_type, seed, label_probability)
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=style["background"])
    RENDERERS[view_type](fig, company, style, np)
    fig.suptitle(
        f"{company['industry'].replace('_', ' ').title()} — {view_type.title()} Metrics",
        fontsize=13,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.11, top=0.88, wspace=0.32, hspace=0.38)
    fig.savefig(destination, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "style": style["name"],
        "show_data_labels": style["show_labels"],
        "width_px": width_px,
        "height_px": height_px,
        "dpi": dpi,
    }
