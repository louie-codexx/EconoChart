from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from econochart.io import write_json

MODEL_TOKEN_ID_FIELDS = (
    "vocab_size",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)
TOKENIZER_CONFIG_FIELDS = (
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "chat_template",
    "added_tokens_decoder",
)
PROCESSOR_FIELDS = (
    "patch_size",
    "temporal_patch_size",
    "merge_size",
    "min_pixels",
    "max_pixels",
    "image_mean",
    "image_std",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required OPD compatibility file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model", {})
    return {
        "model_type": model.get("type") if isinstance(model, dict) else None,
        "vocab": model.get("vocab") if isinstance(model, dict) else None,
        "merges": model.get("merges") if isinstance(model, dict) else None,
        "added_tokens": payload.get("added_tokens"),
        "normalizer": payload.get("normalizer"),
        "pre_tokenizer": payload.get("pre_tokenizer"),
        "post_processor": payload.get("post_processor"),
        "decoder": payload.get("decoder"),
    }


def _selected(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: payload.get(field) for field in fields}


def _compare_field_group(
    name: str,
    student: dict[str, Any],
    teacher: dict[str, Any],
    issues: list[str],
) -> dict[str, Any]:
    mismatched = [field for field in student if student[field] != teacher.get(field)]
    if mismatched:
        issues.append(f"{name} mismatch: {mismatched}")
    return {
        "matched": not mismatched,
        "mismatched_fields": mismatched,
        "student": student,
        "teacher": teacher,
    }


def compare_model_interfaces(student_model_dir: str | Path, teacher_model_dir: str | Path) -> dict[str, Any]:
    student_root = Path(student_model_dir).expanduser().resolve()
    teacher_root = Path(teacher_model_dir).expanduser().resolve()
    if student_root == teacher_root:
        raise ValueError("Student and teacher model directories must be different")
    if not student_root.is_dir() or not teacher_root.is_dir():
        raise FileNotFoundError(f"Student/teacher model directory missing: {student_root}, {teacher_root}")

    issues: list[str] = []
    student_tokenizer = _tokenizer_semantics(_read_json(student_root / "tokenizer.json"))
    teacher_tokenizer = _tokenizer_semantics(_read_json(teacher_root / "tokenizer.json"))
    student_tokenizer_hash = _stable_hash(student_tokenizer)
    teacher_tokenizer_hash = _stable_hash(teacher_tokenizer)
    tokenizer_match = student_tokenizer_hash == teacher_tokenizer_hash
    if not tokenizer_match:
        issues.append("tokenizer semantic fingerprints differ")

    student_config = _read_json(student_root / "config.json")
    teacher_config = _read_json(teacher_root / "config.json")
    student_tokenizer_config = _read_json(student_root / "tokenizer_config.json")
    teacher_tokenizer_config = _read_json(teacher_root / "tokenizer_config.json")
    student_processor = _read_json(student_root / "preprocessor_config.json")
    teacher_processor = _read_json(teacher_root / "preprocessor_config.json")

    model_tokens = _compare_field_group(
        "model token IDs",
        _selected(student_config, MODEL_TOKEN_ID_FIELDS),
        _selected(teacher_config, MODEL_TOKEN_ID_FIELDS),
        issues,
    )
    tokenizer_config = _compare_field_group(
        "tokenizer config",
        _selected(student_tokenizer_config, TOKENIZER_CONFIG_FIELDS),
        _selected(teacher_tokenizer_config, TOKENIZER_CONFIG_FIELDS),
        issues,
    )
    processor_config = _compare_field_group(
        "processor config",
        _selected(student_processor, PROCESSOR_FIELDS),
        _selected(teacher_processor, PROCESSOR_FIELDS),
        issues,
    )

    return {
        "status": "passed" if not issues else "failed",
        "gate": "OPD_MODEL_INTERFACE_COMPATIBILITY_PASS" if not issues else "OPD_MODEL_INTERFACE_COMPATIBILITY_FAIL",
        "student_model_dir": str(student_root),
        "teacher_model_dir": str(teacher_root),
        "tokenizer": {
            "matched": tokenizer_match,
            "student_semantic_sha256": student_tokenizer_hash,
            "teacher_semantic_sha256": teacher_tokenizer_hash,
        },
        "model_tokens": model_tokens,
        "tokenizer_config": tokenizer_config,
        "processor_config": processor_config,
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare student/teacher token and vision interfaces before OPD.")
    parser.add_argument("--student-model-dir", required=True)
    parser.add_argument("--teacher-model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Compatibility report already exists; use --force only after auditing it: {output}")
    report = compare_model_interfaces(args.student_model_dir, args.teacher_model_dir)
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
