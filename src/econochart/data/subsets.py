from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from econochart.config import ConfigError, load_config, project_path, require
from econochart.data.schema import parse_ground_truth
from econochart.io import read_jsonl, sha256_file, stable_seed, write_json, write_jsonl

T = TypeVar("T")


class _FlowEdge:
    __slots__ = ("to", "reverse", "capacity")

    def __init__(self, to: int, reverse: int, capacity: int) -> None:
        self.to = to
        self.reverse = reverse
        self.capacity = capacity


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, capacity: int) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity)
    backward = _FlowEdge(source, len(graph[source]), 0)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _max_flow(graph: list[list[_FlowEdge]], source: int, sink: int) -> int:
    total = 0
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for edge in graph[node]:
                if edge.capacity > 0 and levels[edge.to] < 0:
                    levels[edge.to] = levels[node] + 1
                    queue.append(edge.to)
        if levels[sink] < 0:
            return total

        positions = [0] * len(graph)

        def send(
            node: int,
            amount: int,
            current_levels: list[int],
            current_positions: list[int],
        ) -> int:
            if node == sink:
                return amount
            while current_positions[node] < len(graph[node]):
                edge = graph[node][current_positions[node]]
                if edge.capacity > 0 and current_levels[node] + 1 == current_levels[edge.to]:
                    pushed = send(
                        edge.to,
                        min(amount, edge.capacity),
                        current_levels,
                        current_positions,
                    )
                    if pushed:
                        edge.capacity -= pushed
                        graph[edge.to][edge.reverse].capacity += pushed
                        return pushed
                current_positions[node] += 1
            return 0

        while pushed := send(source, 10**9, levels, positions):
            total += pushed


def _largest_remainder(weights: dict[str, int | float], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    positive = {str(key): float(value) for key, value in weights.items() if float(value) > 0}
    if not positive:
        if total == 0:
            return {str(key): 0 for key in weights}
        raise ValueError("at least one positive weight is required")
    weight_sum = sum(positive.values())
    raw = {key: total * value / weight_sum for key, value in positive.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(positive, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return {str(key): quotas.get(str(key), 0) for key in weights}


def _scenario(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("scenario", "unknown"))


def _stable_order(values: Iterable[T], *, seed: int, namespace: str, key: Callable[[T], str]) -> list[T]:
    return sorted(values, key=lambda value: (stable_seed(f"{namespace}:{key(value)}", seed), key(value)))


def _take_stratified(
    values: Sequence[T],
    count: int,
    *,
    seed: int,
    namespace: str,
    key: Callable[[T], str],
    stratum: Callable[[T], tuple[str, ...]],
) -> list[T]:
    if count < 0 or count > len(values):
        raise ValueError(f"cannot select {count} rows from {len(values)} candidates")
    buckets: dict[tuple[str, ...], list[T]] = defaultdict(list)
    for value in values:
        buckets[stratum(value)].append(value)
    quotas = _largest_remainder({json.dumps(bucket): len(rows) for bucket, rows in buckets.items()}, count)
    selected: list[T] = []
    for bucket, rows in sorted(buckets.items()):
        bucket_key = json.dumps(bucket)
        ordered = _stable_order(rows, seed=seed, namespace=f"{namespace}:{bucket_key}", key=key)
        selected.extend(ordered[: quotas[bucket_key]])
    return _stable_order(selected, seed=seed, namespace=f"{namespace}:output", key=key)


def _assign_one_record_per_group(
    groups: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if sum(quotas.values()) != len(groups):
        raise ValueError("task quotas must sum to the number of groups")
    task_names = sorted(key for key, value in quotas.items() if value > 0)
    ordered_groups = _stable_order(groups, seed=seed, namespace=f"{namespace}:groups", key=str)
    source = 0
    group_offset = 1
    task_offset = group_offset + len(ordered_groups)
    sink = task_offset + len(task_names)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    task_nodes = {task: task_offset + index for index, task in enumerate(task_names)}
    assignments: dict[str, list[tuple[str, _FlowEdge]]] = {}

    for group_index, group_id in enumerate(ordered_groups):
        group_node = group_offset + group_index
        _add_flow_edge(graph, source, group_node, 1)
        rows_by_task = {str(row["task_type"]): row for row in groups[group_id]}
        candidate_tasks = [task for task in task_names if task in rows_by_task]
        candidate_tasks.sort(key=lambda task: (stable_seed(f"{namespace}:{group_id}:{task}", seed), task))
        assignments[group_id] = []
        for task in candidate_tasks:
            edge = _add_flow_edge(graph, group_node, task_nodes[task], 1)
            assignments[group_id].append((task, edge))

    for task in task_names:
        _add_flow_edge(graph, task_nodes[task], sink, int(quotas[task]))

    flow = _max_flow(graph, source, sink)
    if flow != len(groups):
        availability = Counter(
            str(row["task_type"])
            for rows in groups.values()
            for row in {str(value["task_type"]): value for value in rows}.values()
        )
        raise ValueError(
            f"task quotas are infeasible for {namespace}: flow={flow}/{len(groups)}, "
            f"quotas={dict(sorted(quotas.items()))}, availability={dict(sorted(availability.items()))}"
        )

    selected: list[dict[str, Any]] = []
    for group_id in ordered_groups:
        rows_by_task = {str(row["task_type"]): row for row in groups[group_id]}
        used = [task for task, edge in assignments[group_id] if edge.capacity == 0]
        if len(used) != 1:
            raise RuntimeError(f"internal assignment error for group {group_id}: {used}")
        selected.append(rows_by_task[used[0]])
    return selected


def select_sft_records(
    records: Sequence[dict[str, Any]],
    *,
    screen_records_per_chart: int,
    train_records_per_chart: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 1 <= screen_records_per_chart <= train_records_per_chart:
        raise ValueError("SFT records-per-chart values must satisfy 1 <= screen <= train")
    chart_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("split") != "train":
            raise ValueError(f"SFT source contains non-train row: {row.get('id')}")
        chart_groups[str(row["chart_id"])].append(row)
    if not chart_groups:
        raise ValueError("SFT source is empty")
    minimum = min(len(rows) for rows in chart_groups.values())
    if train_records_per_chart > minimum:
        raise ValueError(
            f"requested {train_records_per_chart} SFT rows per chart, but the smallest chart has {minimum}"
        )

    full_task_counts = Counter(str(row["task_type"]) for row in records)
    selected_by_chart: dict[str, list[dict[str, Any]]] = {chart: [] for chart in chart_groups}
    screen_rows: list[dict[str, Any]] = []
    current_task_counts: Counter[str] = Counter()

    for slot in range(train_records_per_chart):
        desired_total = (slot + 1) * len(chart_groups)
        desired = _largest_remainder(dict(full_task_counts), desired_total)
        residual_weights = {
            task: max(0.0, desired.get(task, 0) - current_task_counts.get(task, 0))
            for task in full_task_counts
        }
        slot_quotas = _largest_remainder(residual_weights, len(chart_groups))
        available = {
            chart: [row for row in rows if row["id"] not in {value["id"] for value in selected_by_chart[chart]}]
            for chart, rows in chart_groups.items()
        }
        slot_rows = _assign_one_record_per_group(
            available,
            slot_quotas,
            seed=seed,
            namespace=f"sft:slot:{slot}",
        )
        for row in slot_rows:
            selected_by_chart[str(row["chart_id"])].append(row)
            current_task_counts[str(row["task_type"])] += 1
        if slot + 1 == screen_records_per_chart:
            screen_rows = [value for rows in selected_by_chart.values() for value in rows]

    train_rows = [value for rows in selected_by_chart.values() for value in rows]
    return sorted(screen_rows, key=lambda row: row["id"]), sorted(train_rows, key=lambda row: row["id"])


def _entity_stratum(row: dict[str, Any]) -> tuple[str, ...]:
    return (str(row["industry"]), _scenario(row))


def _select_grpo_chart_ids(
    chart_groups: dict[str, list[dict[str, Any]]],
    prompt_count: int,
    *,
    seed: int,
) -> set[str]:
    entity_charts: dict[str, list[str]] = defaultdict(list)
    chart_rows = {chart: rows[0] for chart, rows in chart_groups.items()}
    for chart, row in chart_rows.items():
        entity_charts[str(row["entity_id"])].append(chart)
    if not len(entity_charts) <= prompt_count <= len(chart_groups):
        raise ValueError(
            f"GRPO prompts must be between entity coverage ({len(entity_charts)}) and chart coverage "
            f"({len(chart_groups)}), got {prompt_count}"
        )
    for entity, charts in entity_charts.items():
        views = [str(chart_rows[chart]["view_type"]) for chart in charts]
        if len(charts) != 2 or views.count("financial") != 1:
            raise ValueError(f"entity {entity} must have exactly one financial and one secondary chart, got {views}")

    view_counts = Counter(str(row["view_type"]) for row in chart_rows.values())
    base_targets = _largest_remainder(dict(view_counts), len(entity_charts))
    final_targets = _largest_remainder(dict(view_counts), prompt_count)
    secondary_views = sorted(view for view in view_counts if view != "financial")
    entities_by_secondary: dict[str, list[str]] = defaultdict(list)
    secondary_chart: dict[str, str] = {}
    financial_chart: dict[str, str] = {}
    for entity, charts in entity_charts.items():
        for chart in charts:
            view = str(chart_rows[chart]["view_type"])
            if view == "financial":
                financial_chart[entity] = chart
            else:
                secondary_chart[entity] = chart
                entities_by_secondary[view].append(entity)

    base_secondary: set[str] = set()
    for view in secondary_views:
        candidates = entities_by_secondary[view]
        count = base_targets.get(view, 0)
        chosen = _take_stratified(
            candidates,
            count,
            seed=seed,
            namespace=f"grpo:base:{view}",
            key=str,
            stratum=lambda entity: _entity_stratum(chart_rows[secondary_chart[entity]]),
        )
        base_secondary.update(chosen)

    selected = {
        secondary_chart[entity] if entity in base_secondary else financial_chart[entity]
        for entity in entity_charts
    }
    for view in secondary_views:
        needed = final_targets.get(view, 0) - base_targets.get(view, 0)
        candidates = [entity for entity in entities_by_secondary[view] if entity not in base_secondary]
        chosen = _take_stratified(
            candidates,
            needed,
            seed=seed,
            namespace=f"grpo:extra:{view}",
            key=str,
            stratum=lambda entity: _entity_stratum(chart_rows[secondary_chart[entity]]),
        )
        selected.update(secondary_chart[entity] for entity in chosen)

    financial_needed = final_targets.get("financial", 0) - base_targets.get("financial", 0)
    financial_candidates = sorted(base_secondary)
    chosen_financial = _take_stratified(
        financial_candidates,
        financial_needed,
        seed=seed,
        namespace="grpo:extra:financial",
        key=str,
        stratum=lambda entity: _entity_stratum(chart_rows[financial_chart[entity]]),
    )
    selected.update(financial_chart[entity] for entity in chosen_financial)
    if len(selected) != prompt_count:
        raise RuntimeError(f"internal GRPO chart selection error: expected {prompt_count}, got {len(selected)}")
    return selected


def select_grpo_records(
    records: Sequence[dict[str, Any]],
    *,
    prompt_count: int,
    task_weights: dict[str, int | float],
    seed: int,
) -> list[dict[str, Any]]:
    chart_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("split") != "train":
            raise ValueError(f"GRPO source contains non-train row: {row.get('id')}")
        if not parse_ground_truth(row["ground_truth"]).get("grpo_eligible", False):
            raise ValueError(f"GRPO source contains ineligible row: {row.get('id')}")
        chart_groups[str(row["chart_id"])].append(row)
    selected_charts = _select_grpo_chart_ids(chart_groups, prompt_count, seed=seed)
    selected_groups = {chart: chart_groups[chart] for chart in selected_charts}
    quotas = _largest_remainder(task_weights, prompt_count)
    rows = _assign_one_record_per_group(selected_groups, quotas, seed=seed, namespace="grpo:tasks")
    return sorted(rows, key=lambda row: row["id"])


def select_validation_records(
    records: Sequence[dict[str, Any]], *, sample_count: int, seed: int
) -> list[dict[str, Any]]:
    if any(row.get("split") != "val" for row in records):
        raise ValueError("validation source contains rows outside split=val")
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        task_groups[str(row["task_type"])].append(row)
    task_quotas = _largest_remainder({task: len(rows) for task, rows in task_groups.items()}, sample_count)
    selected: list[dict[str, Any]] = []
    for task, rows in sorted(task_groups.items()):
        selected.extend(
            _take_stratified(
                rows,
                task_quotas[task],
                seed=seed,
                namespace=f"validation:{task}",
                key=lambda row: str(row["id"]),
                stratum=lambda row: (
                    str(row["industry"]),
                    _scenario(row),
                    str(row["view_type"]),
                    str(row["difficulty"]),
                ),
            )
        )
    return sorted(selected, key=lambda row: row["id"])


def _distribution(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    if field == "scenario":
        values = (_scenario(row) for row in rows)
    else:
        values = (str(row[field]) for row in rows)
    return dict(sorted(Counter(values).items()))


def _subset_summary(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "entities": len({row["entity_id"] for row in rows}),
        "charts": len({row["chart_id"] for row in rows}),
        "task_distribution": _distribution(rows, "task_type"),
        "view_distribution": _distribution(rows, "view_type"),
        "difficulty_distribution": _distribution(rows, "difficulty"),
        "industry_distribution": _distribution(rows, "industry"),
        "scenario_distribution": _distribution(rows, "scenario"),
    }


def build_training_subsets(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    seed = int(config.get("seed", 20260821))
    dataset_root = project_path(require(config, "data.dataset_root"))
    output_dir = project_path(require(config, "data.output_dir"))
    source_manifest = dataset_root / "manifest.json"
    source_paths = {
        "sft_train": dataset_root / "annotations" / "train.jsonl",
        "grpo_train": dataset_root / "annotations" / "grpo_train.jsonl",
        "validation": dataset_root / "annotations" / "val.jsonl",
    }
    required_paths = [source_manifest, *source_paths.values()]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"build and validate EconoChart-v2 before subsets; missing={missing}")

    expected_manifest_sha256 = str(require(config, "data.source_manifest_sha256"))
    actual_manifest_sha256 = sha256_file(source_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "full-dataset manifest does not match the frozen training protocol: "
            f"expected={expected_manifest_sha256}, actual={actual_manifest_sha256}. "
            "Do not mix data versions; investigate or create a new versioned subset config."
        )

    names = {
        "sft_screen": "sft_screen_4800.jsonl",
        "sft_train": "sft_train_9600.jsonl",
        "grpo_train": "grpo_train_3600.jsonl",
        "validation": "val_512.jsonl",
        "manifest": "subset_manifest.json",
    }
    output_paths = {name: output_dir / filename for name, filename in names.items()}
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"subset outputs already exist; pass --force to replace known files: {existing}")

    sft_source = list(read_jsonl(source_paths["sft_train"]))
    grpo_source = list(read_jsonl(source_paths["grpo_train"]))
    validation_source = list(read_jsonl(source_paths["validation"]))
    screen_rows, sft_rows = select_sft_records(
        sft_source,
        screen_records_per_chart=int(require(config, "selection.sft.screen_records_per_chart")),
        train_records_per_chart=int(require(config, "selection.sft.train_records_per_chart")),
        seed=seed,
    )
    grpo_rows = select_grpo_records(
        grpo_source,
        prompt_count=int(require(config, "selection.grpo.prompts")),
        task_weights=dict(require(config, "selection.grpo.task_weights")),
        seed=seed,
    )
    validation_rows = select_validation_records(
        validation_source,
        sample_count=int(require(config, "selection.validation.rows")),
        seed=seed,
    )

    expected = {"sft_screen": 4800, "sft_train": 9600, "grpo_train": 3600, "validation": 512}
    actual = {
        "sft_screen": len(screen_rows),
        "sft_train": len(sft_rows),
        "grpo_train": len(grpo_rows),
        "validation": len(validation_rows),
    }
    if actual != expected:
        raise ValueError(f"final subset sizes do not match the frozen protocol: expected={expected}, actual={actual}")
    if not {row["id"] for row in screen_rows}.issubset({row["id"] for row in sft_rows}):
        raise RuntimeError("SFT screening rows must be nested inside the final SFT subset")
    screen_chart_counts = Counter(str(row["chart_id"]) for row in screen_rows)
    train_chart_counts = Counter(str(row["chart_id"]) for row in sft_rows)
    if set(screen_chart_counts.values()) != {1} or len(screen_chart_counts) != 4800:
        raise RuntimeError("SFT screening subset must contain exactly one row from every train chart")
    if set(train_chart_counts.values()) != {2} or len(train_chart_counts) != 4800:
        raise RuntimeError("final SFT subset must contain exactly two rows from every train chart")
    if len({row["chart_id"] for row in sft_rows}) != 4800 or len({row["entity_id"] for row in sft_rows}) != 2400:
        raise RuntimeError("final SFT subset must cover all 2400 train entities and all 4800 train charts")
    if len({row["chart_id"] for row in grpo_rows}) != 3600:
        raise RuntimeError("GRPO subset must use exactly one prompt from each of 3600 distinct train charts")
    if len({row["entity_id"] for row in grpo_rows}) != 2400:
        raise RuntimeError("GRPO subset must cover all 2400 train entities")
    expected_grpo_tasks = _largest_remainder(
        dict(require(config, "selection.grpo.task_weights")),
        int(require(config, "selection.grpo.prompts")),
    )
    actual_grpo_tasks = Counter(str(row["task_type"]) for row in grpo_rows)
    if dict(sorted(actual_grpo_tasks.items())) != dict(sorted(expected_grpo_tasks.items())):
        raise RuntimeError(
            "GRPO task quotas drifted from the frozen protocol: "
            f"expected={expected_grpo_tasks}, actual={dict(actual_grpo_tasks)}"
        )
    if len({row["id"] for row in validation_rows}) != 512 or any(
        row.get("split") != "val" for row in validation_rows
    ):
        raise RuntimeError("development panel must contain 512 unique validation rows and no test rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "sft_screen": screen_rows,
        "sft_train": sft_rows,
        "grpo_train": grpo_rows,
        "validation": validation_rows,
    }
    for name, rows in payloads.items():
        write_jsonl(output_paths[name], rows)

    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "dataset": source_payload.get("dataset"),
        "source_builder_version": source_payload.get("builder_version"),
        "source_manifest_sha256": actual_manifest_sha256,
        "seed": seed,
        "policy": {
            "sft_screen": "one balanced task per train chart; nested in final SFT",
            "sft_train": "two balanced tasks per train chart; all train entities and charts retained",
            "grpo_train": "all train entities at least once; balanced chart views and explicit task quotas",
            "validation": "512 rows stratified from val only; test is never used for parameter selection",
        },
        "configured_grpo_task_quotas": expected_grpo_tasks,
        "source_checksums": {name: sha256_file(path) for name, path in source_paths.items()},
        "subsets": {name: _subset_summary(output_paths[name], rows) for name, rows in payloads.items()},
        "invariants": {
            "sft_screen_nested_in_sft_train": True,
            "sft_train_all_entities": 2400,
            "sft_train_all_charts": 4800,
            "grpo_train_all_entities": 2400,
            "grpo_train_distinct_charts": 3600,
            "development_rows_from_val": 512,
            "test_rows_used": 0,
        },
    }
    write_json(output_paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic SFT, GRPO, and validation subsets.")
    parser.add_argument("--config", default="configs/data/training_subsets.yaml")
    parser.add_argument("--force", action="store_true", help="Replace only the known subset output files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = build_training_subsets(load_config(args.config), force=args.force)
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
