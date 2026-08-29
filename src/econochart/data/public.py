from __future__ import annotations

import argparse
import base64
import hashlib
import io
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from econochart.config import ROOT, load_config, project_path, require
from econochart.data.schema import make_record
from econochart.io import sha256_file, stable_seed, write_json, write_jsonl

PUBLIC_SOURCES = {
    "chartqa": {
        "dataset_id": "HuggingFaceM4/ChartQA",
        "license": "GPL-3.0",
        "homepage": "https://github.com/vis-nlp/ChartQA",
        "splits": ("train", "val", "test"),
    },
    "chartqapro": {
        "dataset_id": "ahmed-masry/ChartQAPro",
        "license": "MIT",
        "homepage": "https://github.com/vis-nlp/ChartQAPro",
        "splits": ("test",),
    },
}


def _safe_replace_root(path: Path, overwrite: bool) -> None:
    public_root = (ROOT / "data" / "generated" / "public").resolve()
    resolved = path.resolve()
    if resolved == public_root or public_root not in resolved.parents:
        raise ValueError(f"Public dataset output must be a named child below {public_root}: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {resolved}. Pass --overwrite to rebuild it.")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _numeric_answer(value: str) -> float | None:
    normalized = value.strip().replace(",", "").rstrip("%")
    try:
        return float(normalized)
    except ValueError:
        return None


def _public_ground_truth(answers: list[str]) -> dict[str, Any]:
    numeric = _numeric_answer(answers[0]) if answers else None
    numeric_targets = []
    if numeric is not None:
        tolerance = max(0.05, abs(numeric) * 0.05)
        numeric_targets.append(
            {"name": "public_answer", "value": numeric, "unit": "", "tolerance": round(tolerance, 6)}
        )
    return {
        "required_sections": [],
        "numeric_targets": numeric_targets,
        "trend_targets": [],
        "evidence_keywords": [],
        "risk_labels": [],
        "answer_aliases": answers,
        "grpo_eligible": False,
        "length_range": [1, 160],
    }


def _coerce_pil_image(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_file():
            return Image.open(candidate).convert("RGB")
        try:
            return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - report a single actionable conversion error
            raise ValueError("String image is neither a local path nor valid base64 image data") from exc
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


def _save_deduplicated_image(image: Any, images_dir: Path) -> tuple[str, str]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    destination = images_dir / f"{digest[:20]}.png"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return digest, destination.relative_to(ROOT).as_posix()


def _select_indices(length: int, limit: int | None, seed: int, namespace: str) -> list[int]:
    indices = list(range(length))
    if limit is None or limit < 0 or limit >= length:
        return indices
    indices.sort(key=lambda index: stable_seed(f"{namespace}:{index}", seed))
    return sorted(indices[:limit])


def _prepare_chartqa_split(
    dataset: Any,
    split: str,
    output_root: Path,
    *,
    seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    images_dir = output_root / "images" / split
    for source_index in _select_indices(len(dataset), limit, seed, f"chartqa:{split}"):
        source = dataset[source_index]
        image = _coerce_pil_image(source["image"])
        digest, image_path = _save_deduplicated_image(image, images_dir)
        answers = _listify(source.get("label"))
        if not answers:
            continue
        chart_id = f"chartqa_{digest[:16]}"
        question = str(source["query"]).strip()
        record = make_record(
            record_id=f"chartqa_{split}_{source_index:06d}",
            dataset="chartqa",
            split=split,
            entity_id=chart_id,
            chart_id=chart_id,
            image=image_path,
            industry="general",
            view_type="public_chart",
            task_type="public_chartqa",
            difficulty="medium",
            question=question,
            answer=answers[0],
            ground_truth=_public_ground_truth(answers),
            metadata={
                "source_index": source_index,
                "image_sha256": digest,
                "human_or_machine": (
                    int(source["human_or_machine"]) if source.get("human_or_machine") is not None else -1
                ),
                "license": PUBLIC_SOURCES["chartqa"]["license"],
            },
        )
        records.append(record)
    return records


_CHARTQA_SPLITS = ("train", "val", "test")
_CHARTQA_SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def _chartqa_image_sha256(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"ChartQA record {row.get('id', '<unknown>')} has no metadata mapping")
    digest = metadata.get("image_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"ChartQA record {row.get('id', '<unknown>')} has no full image SHA256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(
            f"ChartQA record {row.get('id', '<unknown>')} has an invalid image SHA256"
        ) from exc
    return digest.lower()


def _chartqa_overlap_hashes(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    hashes_by_split = {
        split: {_chartqa_image_sha256(row) for row in rows_by_split[split]}
        for split in _CHARTQA_SPLITS
    }
    return {
        f"{left}_{right}": sorted(hashes_by_split[left] & hashes_by_split[right])
        for left, right in _CHARTQA_SPLIT_PAIRS
    }


def _decontaminate_chartqa_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    missing_splits = [split for split in _CHARTQA_SPLITS if split not in rows_by_split]
    if missing_splits:
        raise ValueError(f"ChartQA rows are missing required splits: {missing_splits}")

    cleaned = {split: list(rows_by_split[split]) for split in _CHARTQA_SPLITS}
    before_records = {split: len(cleaned[split]) for split in _CHARTQA_SPLITS}
    initial_overlap_hashes = _chartqa_overlap_hashes(cleaned)
    if initial_overlap_hashes["val_test"]:
        raise ValueError(
            "ChartQA fixed validation/test splits share "
            f"{len(initial_overlap_hashes['val_test'])} image hashes"
        )

    validation_hashes = {_chartqa_image_sha256(row) for row in cleaned["val"]}
    test_hashes = {_chartqa_image_sha256(row) for row in cleaned["test"]}
    evaluation_hashes = validation_hashes | test_hashes
    kept_train: list[dict[str, Any]] = []
    removed_train_records: list[dict[str, Any]] = []
    for row in cleaned["train"]:
        digest = _chartqa_image_sha256(row)
        if digest not in evaluation_hashes:
            kept_train.append(row)
            continue
        metadata = row["metadata"]
        overlaps_with = []
        if digest in validation_hashes:
            overlaps_with.append("val")
        if digest in test_hashes:
            overlaps_with.append("test")
        removed_train_records.append(
            {
                "id": row.get("id"),
                "source_index": metadata.get("source_index"),
                "image_sha256": digest,
                "overlaps_with": overlaps_with,
            }
        )
    cleaned["train"] = kept_train

    final_overlap_hashes = _chartqa_overlap_hashes(cleaned)
    if any(final_overlap_hashes.values()):
        raise RuntimeError(f"ChartQA split decontamination failed: {final_overlap_hashes}")

    audit = {
        "policy": "preserve_fixed_val_test_remove_overlaps_from_optional_train",
        "reason": "Prevent cross-split image leakage while preserving frozen evaluation splits.",
        "training_cap_refilled": False,
        "before_records": before_records,
        "after_records": {split: len(cleaned[split]) for split in _CHARTQA_SPLITS},
        "initial_overlap_counts": {
            pair: len(hashes) for pair, hashes in initial_overlap_hashes.items()
        },
        "final_overlap_counts": {
            pair: len(hashes) for pair, hashes in final_overlap_hashes.items()
        },
        "initial_overlap_hashes": initial_overlap_hashes,
        "final_overlap_hashes": final_overlap_hashes,
        "removed_train_records": removed_train_records,
        "removed_train_record_ids": [row["id"] for row in removed_train_records],
        "removed_image_sha256": sorted(
            {row["image_sha256"] for row in removed_train_records}
        ),
    }
    return cleaned, audit


def _remove_unreferenced_chartqa_train_images(
    output_root: Path,
    train_rows: list[dict[str, Any]],
) -> list[str]:
    images_dir = (output_root / "images" / "train").resolve()
    resolved_output_root = output_root.resolve()
    if resolved_output_root not in images_dir.parents:
        raise ValueError(f"ChartQA train image directory escapes output root: {images_dir}")
    if not images_dir.is_dir():
        return []

    referenced = {(ROOT / str(row["image"])).resolve() for row in train_rows}
    removed: list[str] = []
    for path in sorted(images_dir.glob("*.png")):
        resolved = path.resolve()
        if images_dir not in resolved.parents or resolved in referenced:
            continue
        path.unlink()
        removed.append(path.relative_to(ROOT).as_posix())
    return removed


def prepare_chartqa(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    from datasets import load_dataset

    seed = int(require(config, "project.seed"))
    source = PUBLIC_SOURCES["chartqa"]
    dataset_config = require(config, "public.chartqa")
    output_root = project_path(dataset_config.get("output_root", "data/generated/public/chartqa"))
    _safe_replace_root(output_root, overwrite)
    limits = dataset_config.get("max_samples", {})
    dataset_id = dataset_config.get("dataset_id", source["dataset_id"])
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in source["splits"]:
        dataset = load_dataset(dataset_id, split=split)
        limit = limits.get(split)
        rows_by_split[split] = _prepare_chartqa_split(
            dataset, split, output_root, seed=seed, limit=limit
        )

    rows_by_split, split_decontamination = _decontaminate_chartqa_rows(rows_by_split)
    split_decontamination["removed_train_image_files"] = (
        _remove_unreferenced_chartqa_train_images(output_root, rows_by_split["train"])
    )
    for split in source["splits"]:
        path = output_root / "annotations" / f"{split}.jsonl"
        counts[split] = write_jsonl(path, rows_by_split[split])
        hashes[f"annotations/{split}.jsonl"] = sha256_file(path)
        print(f"Prepared ChartQA {split}: {counts[split]} records")
    manifest = {
        "dataset": "chartqa",
        "source": {**source, "dataset_id": dataset_id},
        "seed": seed,
        "records": counts,
        "checksums": hashes,
        "split_decontamination": split_decontamination,
        "role": "optional SFT mixture (train/val) and fixed external benchmark (test)",
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def _chartqapro_example(source: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    questions = _listify(source.get("Question"))
    answers = _listify(source.get("Answer"))
    if not questions or not answers:
        raise ValueError("ChartQAPro row has no Question/Answer sequence")
    if len(answers) == 1 and len(questions) > 1:
        answers = answers * len(questions)
    if len(questions) != len(answers):
        raise ValueError(f"ChartQAPro Question/Answer length mismatch: {len(questions)} vs {len(answers)}")
    years = _listify(source.get("Year"))
    if not years:
        years = ["NO"] * len(questions)
    if len(years) == 1 and len(questions) > 1:
        years = years * len(questions)
    if len(years) != len(questions):
        raise ValueError(f"ChartQAPro Question/Year length mismatch: {len(questions)} vs {len(years)}")

    history = [
        f"User: {question.strip()}\nAssistant: {answer.strip()}"
        for question, answer in zip(questions[:-1], answers[:-1])
    ]
    current = questions[-1].strip()
    history_text = "\n".join(history)
    prompt = f"Conversation history:\n{history_text}\nCurrent question: {current}" if history else current
    return prompt, answers[-1].strip(), answers, years


def prepare_chartqapro(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    from datasets import load_dataset

    seed = int(require(config, "project.seed"))
    source_config = PUBLIC_SOURCES["chartqapro"]
    dataset_config = require(config, "public.chartqapro")
    output_root = project_path(dataset_config.get("output_root", "data/generated/public/chartqapro"))
    _safe_replace_root(output_root, overwrite)
    dataset_id = dataset_config.get("dataset_id", source_config["dataset_id"])
    dataset = load_dataset(dataset_id, split="test")
    limit = dataset_config.get("max_samples", {}).get("test")
    source_indices = _select_indices(len(dataset), limit, seed, "chartqapro:test")

    records: list[dict[str, Any]] = []
    images_dir = output_root / "images" / "test"
    type_counts: Counter[str] = Counter()
    for source_index in source_indices:
        source = dataset[source_index]
        image = _coerce_pil_image(source["image"])
        digest, image_path = _save_deduplicated_image(image, images_dir)
        chart_id = f"chartqapro_{digest[:16]}"
        paragraph = source.get("Paragraph")
        paragraph_text = str(paragraph).strip() if paragraph not in (None, "", []) else ""
        question, answer, answer_sequence, year_flags = _chartqapro_example(source)
        prompt = f"Context: {paragraph_text}\nQuestion: {question}" if paragraph_text else question
        question_type = str(source.get("Question Type", "unknown"))
        type_counts[question_type] += 1
        records.append(
            make_record(
                record_id=f"chartqapro_test_{source_index:05d}",
                dataset="chartqapro",
                split="test",
                entity_id=chart_id,
                chart_id=chart_id,
                image=image_path,
                industry="general",
                view_type="public_chart",
                task_type="public_chartqapro",
                difficulty="hard",
                question=prompt,
                answer=answer,
                ground_truth=_public_ground_truth([answer]),
                metadata={
                    "source_index": source_index,
                    "question_type": question_type,
                    "year_flags": year_flags,
                    "answer_sequence": answer_sequence,
                    "license": source_config["license"],
                },
            )
        )
    path = output_root / "annotations" / "test.jsonl"
    write_jsonl(path, records)
    manifest = {
        "dataset": "chartqapro",
        "source": {**source_config, "dataset_id": dataset_id},
        "seed": seed,
        "records": {"test": len(records)},
        "question_types": dict(sorted(type_counts.items())),
        "checksums": {"annotations/test.jsonl": sha256_file(path)},
        "role": "fixed external challenge benchmark only; never used for training or model selection",
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"Prepared ChartQAPro test: {len(records)} records")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize approved public chart datasets.")
    parser.add_argument("--config", default="configs/data/public_datasets.yaml")
    parser.add_argument("--dataset", choices=("chartqa", "chartqapro", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.dataset in {"chartqa", "all"}:
        prepare_chartqa(config, overwrite=args.overwrite)
    if args.dataset in {"chartqapro", "all"}:
        prepare_chartqapro(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
