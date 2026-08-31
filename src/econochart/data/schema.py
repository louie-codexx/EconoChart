from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

DATASET_VERSION = "econochart-v2.0"
SPLITS = {"train", "val", "test"}
TASK_TYPES = {
    "value_retrieval",
    "numerical_reasoning",
    "trend_analysis",
    "relationship_analysis",
    "risk_diagnosis",
    "decision_support",
    "comprehensive_report",
    "public_chartqa",
    "public_chartqapro",
    "public_mmefinance",
}
DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {
    "id",
    "dataset",
    "split",
    "entity_id",
    "chart_id",
    "image",
    "industry",
    "view_type",
    "task_type",
    "difficulty",
    "question",
    "answer",
    "ground_truth",
}


def parse_ground_truth(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError("ground_truth must be a JSON string or object")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("ground_truth JSON must decode to an object")
    return parsed


def encode_ground_truth(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append(f"missing fields: {missing}")
        return errors

    if not isinstance(record["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", record["id"]):
        errors.append("id must be a lowercase, filesystem-safe string")
    if record["split"] not in SPLITS:
        errors.append(f"invalid split: {record['split']!r}")
    if record["task_type"] not in TASK_TYPES:
        errors.append(f"invalid task_type: {record['task_type']!r}")
    if record["difficulty"] not in DIFFICULTIES:
        errors.append(f"invalid difficulty: {record['difficulty']!r}")
    if not isinstance(record["question"], str) or not record["question"].strip():
        errors.append("question must be non-empty")
    if not isinstance(record["answer"], str) or not record["answer"].strip():
        errors.append("answer must be non-empty")

    image = record["image"]
    if not isinstance(image, str) or not image:
        errors.append("image must be a non-empty relative POSIX path")
    else:
        image_path = PurePosixPath(image)
        if image_path.is_absolute() or ".." in image_path.parts or "\\" in image:
            errors.append("image must be a repository-relative POSIX path without '..'")

    try:
        ground_truth = parse_ground_truth(record["ground_truth"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid ground_truth: {exc}")
    else:
        sections = ground_truth.get("required_sections", [])
        if not isinstance(sections, list) or not all(isinstance(value, str) for value in sections):
            errors.append("ground_truth.required_sections must be a list of strings")
        targets = ground_truth.get("numeric_targets", [])
        if not isinstance(targets, list):
            errors.append("ground_truth.numeric_targets must be a list")
        else:
            for index, target in enumerate(targets):
                if not isinstance(target, dict) or not isinstance(target.get("value"), (int, float)):
                    errors.append(f"numeric_targets[{index}] must contain a numeric value")
                    continue
                terms = target.get("terms", [])
                if not isinstance(terms, list) or not all(isinstance(value, str) and value for value in terms):
                    errors.append(f"numeric_targets[{index}].terms must be a list of non-empty strings")
        support = ground_truth.get("numeric_support", [])
        if not isinstance(support, list):
            errors.append("ground_truth.numeric_support must be a list")
        else:
            for index, item in enumerate(support):
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("value"), (int, float))
                    or not isinstance(item.get("tolerance"), (int, float))
                    or item["tolerance"] < 0
                ):
                    errors.append(
                        f"numeric_support[{index}] must contain a numeric value and non-negative tolerance"
                    )
    return errors


def make_record(
    *,
    record_id: str,
    dataset: str,
    split: str,
    entity_id: str,
    chart_id: str,
    image: str,
    industry: str,
    view_type: str,
    task_type: str,
    difficulty: str,
    question: str,
    answer: str,
    ground_truth: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "id": record_id,
        "dataset": dataset,
        "split": split,
        "entity_id": entity_id,
        "chart_id": chart_id,
        "image": image,
        "industry": industry,
        "view_type": view_type,
        "task_type": task_type,
        "difficulty": difficulty,
        "question": question,
        "answer": answer,
        "ground_truth": encode_ground_truth(ground_truth),
        "metadata": metadata or {},
    }
    errors = validate_record(record)
    if errors:
        raise ValueError(f"Invalid record {record_id}: {'; '.join(errors)}")
    return record
