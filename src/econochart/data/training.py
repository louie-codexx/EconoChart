from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from econochart.config import ROOT, project_path
from econochart.data.schema import parse_ground_truth
from econochart.io import read_jsonl, stable_seed

DOMAIN_SYSTEM_PROMPT = (
    "你是一名严谨的数字经济经营分析师。只依据图表中可见信息作答，不得编造数据或因果。"
    "回答应使用题目要求的结构，数据结论必须注明单位；建议必须与已识别的风险对应。"
)

PUBLIC_QA_SYSTEM_PROMPT = (
    "You answer chart questions using only information visible in the chart. "
    "For short-answer questions, return only the concise answer and its unit when applicable."
)

GRPO_SYSTEM_PROMPT = (
    "你是一名严谨的数字经济经营分析师。只依据图表作答。"
    "必须给出【结论】和【数据依据】；只有题目要求时才添加【风险】或【建议】。"
    "不得编造图中不存在的数值。"
)


def _sample_records(
    records: list[dict[str, Any]],
    max_samples: int | None,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples < 0 or max_samples >= len(records):
        return records
    return sorted(
        records,
        key=lambda row: stable_seed(f"{namespace}:{row['id']}", seed),
    )[:max_samples]


def load_record_sources(
    sources: Iterable[dict[str, Any]],
    *,
    expected_split: str,
    seed: int,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        path = project_path(source["path"])
        if not path.is_file():
            if source.get("optional", False):
                print(f"Skipping optional source that is not prepared: {path}")
                continue
            raise FileNotFoundError(f"Training source does not exist: {path}")
        records = list(read_jsonl(path))
        wrong_split = [row["id"] for row in records if row.get("split") != expected_split]
        if wrong_split:
            raise ValueError(
                f"Source {path} contains {len(wrong_split)} rows outside split={expected_split}; first={wrong_split[0]}"
            )
        records = _sample_records(
            records,
            source.get("max_samples"),
            seed,
            namespace=f"source:{source_index}:{path.as_posix()}",
        )
        repeat = int(source.get("repeat", 1))
        if repeat < 1:
            raise ValueError(f"repeat must be >=1 for source {path}")
        combined.extend(records * repeat)
    random.Random(stable_seed(f"shuffle:{expected_split}", seed)).shuffle(combined)
    if not combined:
        raise ValueError(f"No records loaded for split={expected_split}")
    return combined


def _absolute_image_path(record: dict[str, Any]) -> str:
    path = (ROOT / record["image"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Referenced image does not exist for {record['id']}: {path}")
    return str(path)


def _is_public(record: dict[str, Any]) -> bool:
    return str(record.get("dataset", "")).lower() in {"chartqa", "chartqapro"}


def system_prompt_for(record: dict[str, Any]) -> str:
    return PUBLIC_QA_SYSTEM_PROMPT if _is_public(record) else DOMAIN_SYSTEM_PROMPT


def make_sft_example(record: dict[str, Any]) -> dict[str, Any]:
    system_prompt = system_prompt_for(record)
    return {
        "images": [_absolute_image_path(record)],
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": record["question"]},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record["answer"]}],
            }
        ],
    }


def build_sft_dataset(records: list[dict[str, Any]]) -> Any:
    """Create TRL's conversational prompt-completion VLM dataset lazily."""
    from datasets import Dataset, Image, Sequence

    examples = [make_sft_example(record) for record in records]
    dataset = Dataset.from_list(examples)
    return dataset.cast_column("images", Sequence(Image(decode=True)))


def make_grpo_example(record: dict[str, Any]) -> dict[str, Any] | None:
    ground_truth = parse_ground_truth(record["ground_truth"])
    if not ground_truth.get("grpo_eligible", False):
        return None
    return {
        "id": record["id"],
        "prompt": [
            {
                "role": "system",
                "content": [{"type": "text", "text": GRPO_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": record["question"]},
                ],
            },
        ],
        "image": _absolute_image_path(record),
        "ground_truth": record["ground_truth"],
        "task_type": record["task_type"],
    }


def build_grpo_dataset(records: list[dict[str, Any]]) -> Any:
    """Create the `prompt` + `image` format expected by TRL GRPOTrainer."""
    from datasets import Dataset, Image

    examples = []
    for record in records:
        example = make_grpo_example(record)
        if example is not None:
            examples.append(example)
    if not examples:
        raise ValueError("GRPO dataset is empty after filtering grpo_eligible records")
    dataset = Dataset.from_list(examples)
    return dataset.cast_column("image", Image(decode=True))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for record in records:
        datasets[record["dataset"]] = datasets.get(record["dataset"], 0) + 1
        tasks[record["task_type"]] = tasks.get(record["task_type"], 0) + 1
    return {
        "rows": len(records),
        "datasets": dict(sorted(datasets.items())),
        "tasks": dict(sorted(tasks.items())),
    }
