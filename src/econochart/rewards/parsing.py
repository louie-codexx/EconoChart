from __future__ import annotations

import re
from typing import Any

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?")
CLAUSE_PATTERN = re.compile(r"[。！？；;\n]+")


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content
            ).strip()
    if isinstance(completion, list):
        parts = [completion_text(item) for item in completion]
        return "\n".join(part for part in parts if part).strip()
    return str(completion).strip()


def extract_numbers(text: str) -> list[float]:
    normalized = text.replace(",", "").replace("，", "")
    return [float(match.group()) for match in NUMBER_PATTERN.finditer(normalized)]


def clauses(text: str) -> list[str]:
    return [part.strip() for part in CLAUSE_PATTERN.split(text) if part.strip()]


def section_positions(text: str, sections: list[str]) -> list[int]:
    return [text.find(f"【{section}】") for section in sections]
