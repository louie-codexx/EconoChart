from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
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

MMEFINANCE_SYSTEM_PROMPT = "You are a helpful assistant."

GRPO_SYSTEM_PROMPT = (
    "你是一名严谨的数字经济经营分析师。只依据图表作答。"
    "必须给出【结论】和【数据依据】；只有题目要求时才添加【风险】或【建议】。"
    "不得编造图中不存在的数值。"
)

OPD_TEACHER_PROMPT_PROFILE_V2 = "opd_teacher_protocol_v2"

_OPD_TEACHER_TASK_SECTIONS = {
    "value_retrieval": ("结论", "数据依据"),
    "numerical_reasoning": ("结论", "数据依据"),
    "relationship_analysis": ("结论", "数据依据"),
    "trend_analysis": ("结论", "数据依据", "风险"),
    "risk_diagnosis": ("结论", "数据依据", "风险"),
    "decision_support": ("结论", "数据依据", "风险", "建议"),
    "comprehensive_report": ("结论", "数据依据", "风险", "建议"),
}

_OPD_TEACHER_TASK_FOCUS = {
    "value_retrieval": "直接给出被问对象、数值和单位，不扩写未被问的年份或指标。",
    "numerical_reasoning": "写出必要的计算关系和最终结果，明确保留正负号、百分比或百分点单位。",
    "relationship_analysis": "只比较题目点名的指标，明确累计变化、差值、ARPU或利润率变化等目标量。",
    "trend_analysis": "明确判断增强、减弱、分化、上升、下降或稳定，并用累计变化和最新变化支撑。",
    "risk_diagnosis": "只选最重要的两项风险，每项给出题内指标、变化方向和一个量化依据。",
    "decision_support": "建议必须逐项对应已识别风险，并用最少的关键数值解释优先级。",
    "comprehensive_report": "覆盖题目要求的经营结论、关键派生指标、主要风险和可执行建议，避免逐年抄表。",
}


def opd_teacher_required_sections(task_type: str, *, view_type: str | None = None) -> tuple[str, ...]:
    try:
        sections = _OPD_TEACHER_TASK_SECTIONS[task_type]
    except KeyError as error:
        raise ValueError(f"Unsupported OPD teacher task type: {task_type!r}") from error
    if task_type == "relationship_analysis" and view_type == "financial":
        return (*sections, "风险")
    return sections


def _opd_teacher_system_prompt_v2(record: dict[str, Any]) -> str:
    if _is_public(record):
        raise ValueError("OPD teacher protocol v2 is restricted to EconoChart domain records")
    task_type = str(record.get("task_type", ""))
    sections = opd_teacher_required_sections(task_type, view_type=str(record.get("view_type", "")))
    labels = "、".join(f"【{section}】" for section in sections)
    max_chars = 420 if task_type == "comprehensive_report" else 320 if task_type == "decision_support" else 260
    risk_guidance = (
        "【风险】使用收入下降、用户流失、利润率承压、变现能力、获客成本上升、"
        "产品集中或波动等与题意相符的明确表述。"
        if "风险" in sections
        else ""
    )
    recommendation_guidance = "【建议】必须与已写风险逐项对应。" if "建议" in sections else ""
    return (
        "你是一名严谨的数字经济经营分析师。只依据图表和用户问题作答，不得编造数据或因果。"
        "这是机器可审计的严格输出协议："
        f"必须按顺序使用且仅使用以下章节标签，每个标签原样出现一次：{labels}。"
        "标签必须写成全角方括号形式；不得改成Markdown标题、加粗标题或编号标题；标签前不得有开场白。"
        "先识别问题要求的最终指标，再读取或计算所需数值；若问题要求百分比变化、增速差、"
        "利润率变化、CAGR或ARPU，必须明确写出最终值、正负号与单位。"
        "不要用整段原始年度序列代替被问目标，"
        "也不要罗列与结论无关的数字。"
        f"本题任务要求：{_OPD_TEACHER_TASK_FOCUS[task_type]}"
        "【数据依据】中的每个关键数值都要紧邻指标名称。"
        f"{risk_guidance}{recommendation_guidance}"
        f"全文尽量控制在{max_chars}个汉字以内，直接输出答案，不解释协议。"
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


def _sample_task_quotas(
    records: list[dict[str, Any]],
    task_quotas: dict[str, Any],
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if not task_quotas:
        raise ValueError("task_quotas must be a non-empty mapping")
    selected: list[dict[str, Any]] = []
    for task_type, raw_quota in sorted(task_quotas.items()):
        quota = int(raw_quota)
        if quota < 1:
            raise ValueError(f"task_quotas[{task_type!r}] must be positive")
        candidates = [row for row in records if str(row.get("task_type")) == str(task_type)]
        if len(candidates) < quota:
            raise ValueError(
                f"task_quotas[{task_type!r}] requires {quota} rows, found {len(candidates)}"
            )
        ranked = sorted(
            candidates,
            key=lambda row: stable_seed(f"{namespace}:task:{task_type}:{row['id']}", seed),
        )
        selected.extend(ranked[:quota])
    return selected


def _sampling_namespace(source: dict[str, Any], source_index: int) -> str:
    explicit = source.get("sampling_namespace")
    if explicit is not None:
        value = str(explicit).strip()
        if not value:
            raise ValueError("sampling_namespace must be a non-empty string")
        return f"source:{value}"
    configured_path = Path(str(source["path"])).as_posix()
    return f"source:{source_index}:{configured_path}"


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
        namespace = _sampling_namespace(source, source_index)
        task_quotas = source.get("task_quotas")
        if task_quotas is not None:
            if source.get("max_samples") is not None:
                raise ValueError("A source cannot configure both max_samples and task_quotas")
            if not isinstance(task_quotas, dict):
                raise ValueError("task_quotas must be a mapping")
            records = _sample_task_quotas(records, task_quotas, seed, namespace)
        else:
            records = _sample_records(
                records,
                source.get("max_samples"),
                seed,
                namespace=namespace,
            )
        expected_rows = source.get("expected_rows")
        if expected_rows is not None and len(records) != int(expected_rows):
            raise ValueError(
                f"Source {path} expected {int(expected_rows)} effective rows, found {len(records)}. "
                "Rebuild the frozen training subsets before launching this run."
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
    return str(record.get("dataset", "")).lower() in {"chartqa", "chartqapro", "mmefinance"}


def system_prompt_for(record: dict[str, Any], profile: str | None = None) -> str:
    if profile == OPD_TEACHER_PROMPT_PROFILE_V2:
        return _opd_teacher_system_prompt_v2(record)
    if profile is not None:
        raise ValueError(f"Unsupported system prompt profile: {profile!r}")
    if str(record.get("dataset", "")).lower() == "mmefinance":
        return MMEFINANCE_SYSTEM_PROMPT
    return PUBLIC_QA_SYSTEM_PROMPT if _is_public(record) else DOMAIN_SYSTEM_PROMPT


def user_prompt_for(record: dict[str, Any]) -> str:
    question = str(record["question"])
    if str(record.get("dataset", "")).lower() != "mmefinance":
        return question
    background = str(record.get("metadata", {}).get("background", "")).strip()
    if not background:
        return question
    return f"Background:\n{background}\n\nQuestion:\n{question}"


def make_sft_example(record: dict[str, Any]) -> dict[str, Any]:
    system_prompt = system_prompt_for(record)
    user_prompt = user_prompt_for(record)
    return {
        "images": [_absolute_image_path(record)],
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
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
