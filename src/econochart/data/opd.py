from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from econochart.config import load_config, project_path, require
from econochart.data.schema import validate_record
from econochart.data.subsets import _largest_remainder, _scenario, _take_stratified
from econochart.io import read_jsonl, sha256_file, stable_seed, write_json, write_jsonl

OPD_DATASET_VERSION = "econochart-opd-v1"


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _records_sha256(records: Sequence[dict[str, Any]]) -> str:
    return _sha256_lines(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )


def _distribution(records: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    if field == "scenario":
        values = (_scenario(record) for record in records)
    else:
        values = (str(record.get(field, "unknown")) for record in records)
    return dict(sorted(Counter(values).items()))


def _load_source(source: dict[str, Any], *, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = project_path(require({"source": source}, "source.path"))
    optional = bool(source.get("optional", False))
    if not path.is_file():
        if optional:
            return [], {"path": str(path), "exists": False, "optional": True, "rows": 0}
        raise FileNotFoundError(f"Required OPD {label} source does not exist: {path}")
    records = list(read_jsonl(path))
    expected_rows = source.get("expected_rows")
    if expected_rows is not None and len(records) != int(expected_rows):
        raise ValueError(f"OPD {label} source {path} expected {int(expected_rows)} rows, found {len(records)}")
    expected_split = source.get("expected_split")
    wrong_split = [str(row.get("id", "<missing>")) for row in records if row.get("split") != expected_split]
    if expected_split is not None and wrong_split:
        raise ValueError(
            f"OPD {label} source {path} contains {len(wrong_split)} rows outside split={expected_split}; "
            f"first={wrong_split[0]}"
        )
    schema_errors = [
        f"{row.get('id', '<missing>')}: {error}"
        for row in records
        for error in validate_record(row)
    ]
    if schema_errors:
        raise ValueError(f"OPD {label} source {path} failed schema validation: {schema_errors[:3]}")
    return records, {
        "path": str(path),
        "exists": True,
        "optional": optional,
        "rows": len(records),
        "sha256": sha256_file(path),
        "ordered_ids_sha256": _sha256_lines(str(row["id"]) for row in records),
        "ordered_records_sha256": _records_sha256(records),
    }


def _validate_unique_ids(records: Sequence[dict[str, Any]], *, label: str) -> None:
    counts = Counter(str(row["id"]) for row in records)
    duplicates = sorted(record_id for record_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"OPD {label} contains duplicate IDs; first={duplicates[:3]}")


def select_opd_records(
    records: Sequence[dict[str, Any]],
    *,
    excluded_ids: set[str],
    round_rows: Sequence[int],
    task_weights: dict[str, int | float],
    seed: int,
    max_records_per_chart: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select unseen train prompts with exact task quotas and coverage-first chart use."""
    if len(round_rows) != 2 or any(int(value) <= 0 for value in round_rows):
        raise ValueError("selection.round_rows must contain two positive integers")
    if max_records_per_chart < 1:
        raise ValueError("max_records_per_chart must be positive")
    _validate_unique_ids(records, label="train source")
    candidates = [row for row in records if str(row["id"]) not in excluded_ids]
    if any(row.get("split") != "train" for row in candidates):
        raise ValueError("OPD candidate source contains records outside split=train")
    total = sum(int(value) for value in round_rows)
    if len(candidates) < total:
        raise ValueError(f"OPD requires {total} unseen prompts, but only {len(candidates)} remain")

    task_quotas = _largest_remainder(task_weights, total)
    available_tasks = Counter(str(row["task_type"]) for row in candidates)
    infeasible = {
        task: {"required": quota, "available": available_tasks.get(task, 0)}
        for task, quota in task_quotas.items()
        if available_tasks.get(task, 0) < quota
    }
    if infeasible:
        raise ValueError(f"OPD task quotas are infeasible: {infeasible}")

    heaps: dict[str, list[tuple[int, int, str, dict[str, Any]]]] = {task: [] for task in task_quotas}
    for row in candidates:
        task = str(row["task_type"])
        if task not in heaps:
            continue
        record_id = str(row["id"])
        priority = stable_seed(f"opd:candidate:{task}:{record_id}", seed)
        heapq.heappush(heaps[task], (0, priority, record_id, row))

    selected: list[dict[str, Any]] = []
    selected_per_task: Counter[str] = Counter()
    chart_counts: Counter[str] = Counter()
    for step in range(total):
        remaining_tasks = [task for task, quota in task_quotas.items() if selected_per_task[task] < quota]
        task = min(
            remaining_tasks,
            key=lambda value: (
                selected_per_task[value] / task_quotas[value],
                stable_seed(f"opd:schedule:{step}:{value}", seed),
                value,
            ),
        )
        row: dict[str, Any] | None = None
        while heaps[task]:
            recorded_count, priority, record_id, candidate = heapq.heappop(heaps[task])
            chart_id = str(candidate["chart_id"])
            current_count = chart_counts[chart_id]
            if current_count >= max_records_per_chart:
                continue
            if recorded_count != current_count:
                heapq.heappush(heaps[task], (current_count, priority, record_id, candidate))
                continue
            row = candidate
            break
        if row is None:
            raise ValueError(
                f"OPD chart cap makes task quota infeasible for {task}; selected={selected_per_task[task]}, "
                f"required={task_quotas[task]}, max_records_per_chart={max_records_per_chart}"
            )
        selected.append(row)
        selected_per_task[task] += 1
        chart_counts[str(row["chart_id"])] += 1

    first_count = int(round_rows[0])
    first = _take_stratified(
        selected,
        first_count,
        seed=seed,
        namespace="opd:round1",
        key=lambda row: str(row["id"]),
        stratum=lambda row: (
            str(row["task_type"]),
            str(row["difficulty"]),
            str(row["industry"]),
            str(row["view_type"]),
            _scenario(row),
        ),
    )
    first_ids = {str(row["id"]) for row in first}
    second = [row for row in selected if str(row["id"]) not in first_ids]
    return sorted(first, key=lambda row: str(row["id"])), sorted(second, key=lambda row: str(row["id"]))


def _dual_rollout_ids(records: Sequence[dict[str, Any]], *, fraction: float, seed: int, namespace: str) -> set[str]:
    if not 0 <= fraction <= 1:
        raise ValueError("dual_rollout_fraction must be between 0 and 1")
    count = round(len(records) * fraction)
    hard = [row for row in records if row.get("difficulty") == "hard"]
    if len(hard) >= count:
        selected = _take_stratified(
            hard,
            count,
            seed=seed,
            namespace=f"{namespace}:hard",
            key=lambda row: str(row["id"]),
            stratum=lambda row: (str(row["task_type"]), str(row["view_type"])),
        )
    else:
        selected = list(hard)
        selected_ids = {str(row["id"]) for row in selected}
        remainder = [row for row in records if str(row["id"]) not in selected_ids]
        selected.extend(
            _take_stratified(
                remainder,
                count - len(selected),
                seed=seed,
                namespace=f"{namespace}:remainder",
                key=lambda row: str(row["id"]),
                stratum=lambda row: (str(row["task_type"]), str(row["difficulty"])),
            )
        )
    return {str(row["id"]) for row in selected}


def _annotate_round(
    records: Sequence[dict[str, Any]],
    *,
    round_id: int,
    dual_rollout_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    dual_ids = _dual_rollout_ids(
        records,
        fraction=dual_rollout_fraction,
        seed=seed,
        namespace=f"opd:round{round_id}:rollouts",
    )
    annotated: list[dict[str, Any]] = []
    for record in records:
        row = copy.deepcopy(record)
        metadata = dict(row.get("metadata", {}))
        metadata["opd"] = {
            "dataset_version": OPD_DATASET_VERSION,
            "round": round_id,
            "rollout_count": 2 if str(row["id"]) in dual_ids else 1,
            "original_answer_role": "audit_only_not_training_target",
        }
        row["metadata"] = metadata
        annotated.append(row)
    return annotated


def _image_identity(
    record: dict[str, Any],
    *,
    verify_image_content: bool,
    image_hash_cache: dict[Path, str],
) -> str:
    image_value = str(record["image"])
    if not verify_image_content:
        return f"path:{image_value}"
    path = project_path(image_value)
    if not path.is_file():
        raise FileNotFoundError(f"OPD image content audit cannot find {record['id']}: {path}")
    if path not in image_hash_cache:
        image_hash_cache[path] = sha256_file(path)
    return f"sha256:{image_hash_cache[path]}"


def split_disjoint_image_panels(
    records: Sequence[dict[str, Any]],
    *,
    first_rows: int,
    seed: int,
    verify_image_content: bool,
    image_hash_cache: dict[Path, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create exact-size panels without placing one image (or image hash) in both."""
    if not 0 < first_rows < len(records):
        raise ValueError("first_rows must leave at least one record in each image-disjoint panel")
    cache = {} if image_hash_cache is None else image_hash_cache
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        identity = _image_identity(
            record,
            verify_image_content=verify_image_content,
            image_hash_cache=cache,
        )
        groups.setdefault(identity, []).append(record)

    ordered_groups = sorted(
        groups,
        key=lambda value: (stable_seed(f"opd:qualification-panel:{value}", seed), value),
    )
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for identity in ordered_groups:
        size = len(groups[identity])
        for current, chosen in sorted(list(reachable.items()), reverse=True):
            candidate = current + size
            if candidate <= first_rows and candidate not in reachable:
                reachable[candidate] = (*chosen, identity)
    if first_rows not in reachable:
        sizes = sorted(Counter(len(group) for group in groups.values()).items())
        raise ValueError(
            f"Cannot form an exact {first_rows}-row image-disjoint qualification panel; group_sizes={sizes}"
        )
    first_identities = set(reachable[first_rows])
    first = [record for identity in first_identities for record in groups[identity]]
    second = [record for identity in groups if identity not in first_identities for record in groups[identity]]
    return sorted(first, key=lambda row: str(row["id"])), sorted(second, key=lambda row: str(row["id"]))


def assert_no_forbidden_overlap(
    selected: Sequence[dict[str, Any]],
    forbidden: Sequence[dict[str, Any]],
    *,
    verify_image_content: bool = False,
    image_hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    selected_ids = {str(row["id"]) for row in selected}
    selected_images = {str(row["image"]) for row in selected}
    forbidden_ids = {str(row["id"]) for row in forbidden}
    forbidden_images = {str(row["image"]) for row in forbidden}
    id_overlap = sorted(selected_ids & forbidden_ids)
    image_overlap = sorted(selected_images & forbidden_images)
    image_sha256_overlap: list[str] = []
    if verify_image_content:
        cache = {} if image_hash_cache is None else image_hash_cache
        selected_hashes = {
            _image_identity(row, verify_image_content=True, image_hash_cache=cache)
            for row in selected
        }
        forbidden_hashes = {
            _image_identity(row, verify_image_content=True, image_hash_cache=cache)
            for row in forbidden
        }
        image_sha256_overlap = sorted(selected_hashes & forbidden_hashes)
    if id_overlap or image_overlap or image_sha256_overlap:
        raise ValueError(
            "OPD selection overlaps isolated data: "
            f"ids={id_overlap[:3]}, image_paths={image_overlap[:3]}, "
            f"image_sha256={image_sha256_overlap[:3]}"
        )
    return {
        "id_overlap_count": 0,
        "image_path_overlap_count": 0,
        "image_sha256_overlap_count": 0,
        "image_content_checked": verify_image_content,
    }


def _summary(path: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rollout_counts = Counter(int(row.get("metadata", {}).get("opd", {}).get("rollout_count", 1)) for row in records)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(records),
        "unique_ids": len({str(row["id"]) for row in records}),
        "entities": len({str(row["entity_id"]) for row in records}),
        "charts": len({str(row["chart_id"]) for row in records}),
        "unique_image_paths": len({str(row["image"]) for row in records}),
        "ordered_ids_sha256": _sha256_lines(str(row["id"]) for row in records),
        "ordered_records_sha256": _records_sha256(records),
        "task_distribution": _distribution(records, "task_type"),
        "difficulty_distribution": _distribution(records, "difficulty"),
        "view_distribution": _distribution(records, "view_type"),
        "industry_distribution": _distribution(records, "industry"),
        "scenario_distribution": _distribution(records, "scenario"),
        "rollout_count_distribution": {str(key): value for key, value in sorted(rollout_counts.items())},
    }


def build_opd_subsets(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    seed = int(config.get("seed", 20260821))
    dataset_root = project_path(require(config, "data.dataset_root"))
    source_manifest_path = dataset_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"EconoChart source manifest is missing: {source_manifest_path}")
    expected_manifest = str(require(config, "data.source_manifest_sha256"))
    actual_manifest = sha256_file(source_manifest_path)
    if actual_manifest != expected_manifest:
        raise ValueError(
            f"EconoChart source manifest changed: expected {expected_manifest}, found {actual_manifest}"
        )

    train, train_source = _load_source(require(config, "data.train_source"), label="train")
    seen: list[dict[str, Any]] = []
    seen_sources: list[dict[str, Any]] = []
    for index, source in enumerate(require(config, "data.seen_sources")):
        rows, summary = _load_source(source, label=f"seen[{index}]")
        summary["reason"] = source.get("reason")
        seen.extend(rows)
        seen_sources.append(summary)
    qualification, qualification_source = _load_source(
        require(config, "data.qualification_source"), label="qualification"
    )
    forbidden: list[dict[str, Any]] = []
    forbidden_sources: list[dict[str, Any]] = []
    for index, source in enumerate(require(config, "data.forbidden_evaluation_sources")):
        rows, summary = _load_source(source, label=f"forbidden[{index}]")
        summary["role"] = source.get("role")
        forbidden.extend(rows)
        forbidden_sources.append(summary)

    seen_ids = {str(row["id"]) for row in seen}
    selection = require(config, "selection")
    round_1, round_2 = select_opd_records(
        train,
        excluded_ids=seen_ids,
        round_rows=selection["round_rows"],
        task_weights=selection["task_weights"],
        seed=seed,
        max_records_per_chart=int(selection["max_records_per_chart"]),
    )
    round_1 = _annotate_round(
        round_1,
        round_id=1,
        dual_rollout_fraction=float(selection["dual_rollout_fraction"]),
        seed=seed,
    )
    round_2 = _annotate_round(
        round_2,
        round_id=2,
        dual_rollout_fraction=float(selection["dual_rollout_fraction"]),
        seed=seed,
    )
    all_training = [*round_1, *round_2]
    verify_image_content = bool(selection.get("verify_image_content", False))
    image_hash_cache: dict[Path, str] = {}
    leakage = assert_no_forbidden_overlap(
        all_training,
        forbidden,
        verify_image_content=verify_image_content,
        image_hash_cache=image_hash_cache,
    )
    training_to_qualification = assert_no_forbidden_overlap(
        all_training,
        qualification,
        verify_image_content=verify_image_content,
        image_hash_cache=image_hash_cache,
    )

    qualification_rows = int(selection["qualification_rows"])
    development_rows = int(selection["development_rows"])
    if qualification_rows + development_rows != len(qualification):
        raise ValueError(
            "qualification_rows + development_rows must consume the frozen qualification source exactly"
        )
    teacher_qualification, development = split_disjoint_image_panels(
        qualification,
        first_rows=qualification_rows,
        seed=seed,
        verify_image_content=verify_image_content,
        image_hash_cache=image_hash_cache,
    )
    if len(development) != development_rows:
        raise ValueError(f"OPD development panel expected {development_rows} rows, found {len(development)}")
    qualification_to_development = assert_no_forbidden_overlap(
        teacher_qualification,
        development,
        verify_image_content=verify_image_content,
        image_hash_cache=image_hash_cache,
    )

    output_dir = project_path(require(config, "data.output_dir"))
    artifacts = require(config, "artifacts")
    paths = {
        "round_1": output_dir / artifacts["round_1"],
        "round_2": output_dir / artifacts["round_2"],
        "teacher_qualification": output_dir / artifacts["teacher_qualification"],
        "development": output_dir / artifacts["development"],
        "manifest": output_dir / artifacts["manifest"],
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"OPD outputs already exist; use --force only after auditing them: {existing}")
    write_jsonl(paths["round_1"], round_1)
    write_jsonl(paths["round_2"], round_2)
    write_jsonl(paths["teacher_qualification"], teacher_qualification)
    write_jsonl(paths["development"], development)

    manifest = {
        "dataset_version": OPD_DATASET_VERSION,
        "experiment_id": config.get("experiment", {}).get("id"),
        "seed": seed,
        "role": "training-only OPD prompts plus development-only teacher qualification; never final evaluation",
        "source_manifest": {"path": str(source_manifest_path), "sha256": actual_manifest},
        "sources": {
            "train": train_source,
            "seen": seen_sources,
            "qualification": qualification_source,
            "forbidden_evaluation": forbidden_sources,
        },
        "selection": {
            "available_train_rows": len(train),
            "seen_row_occurrences": len(seen),
            "seen_unique_ids": len(seen_ids),
            "unseen_candidate_rows": sum(str(row["id"]) not in seen_ids for row in train),
            "round_rows": list(selection["round_rows"]),
            "round_2_policy": selection.get("round_2_policy", "conditional"),
            "max_records_per_chart": int(selection["max_records_per_chart"]),
            "dual_rollout_fraction": float(selection["dual_rollout_fraction"]),
            "verify_image_content": verify_image_content,
            "qualification_panel_grouping": "image_sha256" if verify_image_content else "image_path",
            "task_weights": selection["task_weights"],
        },
        "leakage": leakage,
        "training_to_qualification_leakage": training_to_qualification,
        "qualification_to_development_leakage": qualification_to_development,
        "artifacts": {
            "round_1": _summary(paths["round_1"], round_1),
            "round_2": _summary(paths["round_2"], round_2),
            "teacher_qualification": _summary(paths["teacher_qualification"], teacher_qualification),
            "development": _summary(paths["development"], development),
        },
        "gates": {
            "train_only": all(row["split"] == "train" for row in all_training),
            "seen_id_overlap_count": len({str(row["id"]) for row in all_training} & seen_ids),
            "round_id_overlap_count": len(
                {str(row["id"]) for row in round_1} & {str(row["id"]) for row in round_2}
            ),
            "qualification_to_development_id_overlap_count": len(
                {str(row["id"]) for row in teacher_qualification}
                & {str(row["id"]) for row in development}
            ),
            "qualification_to_development_image_path_overlap_count": len(
                {str(row["image"]) for row in teacher_qualification}
                & {str(row["image"]) for row in development}
            ),
            "forbidden_evaluation_overlap": leakage,
            "round_2_requires_round_1_gate": True,
        },
        "policy": config.get("policy", {}),
    }
    write_json(paths["manifest"], manifest)
    manifest["manifest_sha256"] = sha256_file(paths["manifest"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen training-only prompts for asynchronous OPD.")
    parser.add_argument("--config", default="configs/data/opd_v1.yaml")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_opd_subsets(load_config(args.config), force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
