from __future__ import annotations

import argparse
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from econochart import __version__
from econochart.config import ROOT, load_config, project_path, require
from econochart.data.charts import render_chart
from econochart.data.generator import generate_companies
from econochart.data.questions import build_records_for_chart
from econochart.data.schema import DATASET_VERSION, parse_ground_truth
from econochart.io import sha256_file, write_json, write_jsonl


def _assert_safe_output_root(path: Path) -> None:
    root = ROOT.resolve()
    resolved = path.resolve()
    data_root = (root / "data").resolve()
    if resolved == root or resolved == data_root:
        raise ValueError(f"Refusing to use broad output directory: {resolved}")
    if data_root not in resolved.parents:
        raise ValueError(
            f"Generated data must live below {data_root}. Place the repository itself on the intended data disk."
        )


def _render_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata = render_chart(
        job["company"],
        job["view_type"],
        job["output_path"],
        seed=job["seed"],
        width_px=job["width_px"],
        height_px=job["height_px"],
        dpi=job["dpi"],
        label_probability=job["label_probability"],
    )
    return job["chart_id"], metadata


def _render_all(jobs: list[dict[str, Any]], workers: int) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            chart_id, chart_metadata = _render_job(job)
            metadata[chart_id] = chart_metadata
            if index % 100 == 0 or index == len(jobs):
                print(f"Rendered {index}/{len(jobs)} charts")
        return metadata

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_render_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            chart_id, chart_metadata = future.result()
            metadata[chart_id] = chart_metadata
            if index % 100 == 0 or index == len(jobs):
                print(f"Rendered {index}/{len(jobs)} charts")
    return metadata


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def build_dataset(
    config: dict[str, Any],
    *,
    num_companies_override: int | None = None,
    output_root_override: str | None = None,
    overwrite: bool = False,
    skip_images: bool = False,
    workers_override: int | None = None,
) -> dict[str, Any]:
    data_config = require(config, "data")
    seed = int(require(config, "project.seed"))
    num_companies = int(num_companies_override or require(config, "data.num_companies"))
    output_root = project_path(output_root_override or require(config, "data.output_root"))
    _assert_safe_output_root(output_root)

    if skip_images and overwrite:
        raise ValueError("--skip-images and --overwrite are mutually exclusive because overwrite removes images")
    if output_root.exists() and any(output_root.iterdir()):
        if skip_images:
            print(f"Preserving existing images and rebuilding deterministic metadata under {output_root}")
        elif not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. Pass --overwrite only when you intend to rebuild it."
            )
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    companies = generate_companies(
        num_companies=num_companies,
        seed=seed,
        split_ratios={key: float(value) for key, value in require(config, "data.split_ratios").items()},
    )
    companies_path = output_root / "raw" / "companies.jsonl"
    write_jsonl(companies_path, companies)

    render_config = data_config.get("render", {})
    width_px = int(render_config.get("width_px", 1280))
    height_px = int(render_config.get("height_px", 768))
    dpi = int(render_config.get("dpi", 160))
    label_probability = float(render_config.get("label_probability", 0.5))
    workers = int(workers_override if workers_override is not None else render_config.get("workers", 4))

    jobs: list[dict[str, Any]] = []
    chart_paths: dict[str, str] = {}
    for company in companies:
        for view_type in company["views"]:
            chart_id = f"{company['entity_id']}_{view_type}"
            absolute_path = output_root / "images" / company["split"] / f"{chart_id}.png"
            relative_path = absolute_path.relative_to(ROOT).as_posix()
            chart_paths[chart_id] = relative_path
            jobs.append(
                {
                    "company": company,
                    "view_type": view_type,
                    "chart_id": chart_id,
                    "output_path": str(absolute_path),
                    "seed": seed,
                    "width_px": width_px,
                    "height_px": height_px,
                    "dpi": dpi,
                    "label_probability": label_probability,
                }
            )

    if skip_images:
        missing = [job["output_path"] for job in jobs if not Path(job["output_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"--skip-images was set, but {len(missing)} chart files are missing; first: {missing[0]}")
        chart_metadata = {
            job["chart_id"]: {
                "style": "existing",
                "show_data_labels": None,
                "width_px": width_px,
                "height_px": height_px,
                "dpi": dpi,
            }
            for job in jobs
        }
    else:
        chart_metadata = _render_all(jobs, workers=workers)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for company in companies:
        for view_type in company["views"]:
            chart_id = f"{company['entity_id']}_{view_type}"
            rows = build_records_for_chart(company, view_type, chart_paths[chart_id], seed)
            for row in rows:
                row["metadata"]["chart_style"] = chart_metadata[chart_id]
            rows_by_split[company["split"]].extend(rows)

    annotation_paths: dict[str, Path] = {}
    for split, rows in rows_by_split.items():
        path = output_root / "annotations" / f"{split}.jsonl"
        write_jsonl(path, rows)
        annotation_paths[split] = path

    grpo_rows = [
        row
        for row in rows_by_split["train"]
        if parse_ground_truth(row["ground_truth"]).get("grpo_eligible", False)
    ]
    grpo_path = output_root / "annotations" / "grpo_train.jsonl"
    write_jsonl(grpo_path, grpo_rows)

    split_entities = {
        split: sorted(company["entity_id"] for company in companies if company["split"] == split)
        for split in ("train", "val", "test")
    }
    write_json(output_root / "manifests" / "split_entities.json", split_entities)

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    manifest = {
        "dataset": DATASET_VERSION,
        "builder_version": __version__,
        "seed": seed,
        "num_companies": len(companies),
        "num_charts": len(jobs),
        "num_records": len(all_rows),
        "num_grpo_records": len(grpo_rows),
        "split_entities": {key: len(value) for key, value in split_entities.items()},
        "split_records": {key: len(value) for key, value in rows_by_split.items()},
        "industry_distribution": _distribution(companies, "industry"),
        "scenario_distribution": _distribution(companies, "scenario"),
        "task_distribution": _distribution(all_rows, "task_type"),
        "view_distribution": _distribution(all_rows, "view_type"),
        "difficulty_distribution": _distribution(all_rows, "difficulty"),
        "render": {
            "width_px": width_px,
            "height_px": height_px,
            "dpi": dpi,
            "label_probability": label_probability,
        },
        "checksums": {
            **{f"annotations/{split}.jsonl": sha256_file(path) for split, path in annotation_paths.items()},
            "annotations/grpo_train.jsonl": sha256_file(grpo_path),
            "raw/companies.jsonl": sha256_file(companies_path),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        f"Built {manifest['dataset']}: {manifest['num_companies']} companies, "
        f"{manifest['num_charts']} charts, {manifest['num_records']} records."
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic EconoChart-v2 data.")
    parser.add_argument("--config", default="configs/data/econochart_v2.yaml")
    parser.add_argument("--num-companies", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_dataset(
        load_config(args.config),
        num_companies_override=args.num_companies,
        output_root_override=args.output_root,
        overwrite=args.overwrite,
        skip_images=args.skip_images,
        workers_override=args.workers,
    )


if __name__ == "__main__":
    main()
