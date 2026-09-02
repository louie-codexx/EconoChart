from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from econochart.io import read_json, sha256_file

ROLLOUT_SCHEMA_VERSION = "econochart-opd-rollout-v1"
TEACHER_SCORE_SCHEMA_VERSION = "econochart-opd-teacher-score-v1"


def validate_adapter_snapshot(adapter_path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Bind OPD to one immutable PEFT safetensors adapter."""
    root = Path(adapter_path).expanduser().resolve()
    weights = root / "adapter_model.safetensors"
    config = root / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f"OPD adapter snapshot is incomplete: {root}")
    actual_sha256 = sha256_file(weights)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"OPD adapter SHA256 differs: expected {expected_sha256}, found {actual_sha256}"
        )
    return {
        "path": str(root),
        "weights": str(weights),
        "weights_sha256": actual_sha256,
        "config_sha256": sha256_file(config),
    }


def validate_resume_snapshot(
    output_dir: str | Path,
    *,
    stage: str,
    config: dict[str, Any],
    model_path: str,
    adapter_path: str | None,
    data_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require an interrupted OPD run to match its original immutable identity."""
    path = Path(output_dir).resolve() / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume OPD without the original run manifest: {path}")
    snapshot = read_json(path)
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    if snapshot.get("stage") != stage:
        raise ValueError(f"OPD resume stage differs: expected {stage}, found {snapshot.get('stage')}")
    if snapshot.get("model") != {"base": model_path, "adapter": adapter_path}:
        raise ValueError("OPD resume model or adapter differs from the original run")
    if snapshot.get("config") != clean_config:
        raise ValueError("OPD resume config differs from the original run")
    if data_summary is not None and snapshot.get("data") != data_summary:
        raise ValueError("OPD resume inputs or prerequisite artifacts differ from the original run")
    return snapshot


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_identity(record: dict[str, Any]) -> str:
    return sha256_json(
        {
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "split": record.get("split"),
            "image": record.get("image"),
            "question": record.get("question"),
            "ground_truth": record.get("ground_truth"),
        }
    )


def validate_rollout_row(row: dict[str, Any]) -> None:
    if row.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported rollout schema: {row.get('schema_version')!r}")
    if row.get("split") != "train":
        raise ValueError("OPD rollout rows must come from split=train")
    for field in ("rollout_key", "record_id", "record_sha256", "image"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"OPD rollout field {field!r} must be a non-empty string")
    if not isinstance(row.get("prediction"), str):
        raise ValueError("OPD rollout field 'prediction' must be a string")
    if int(row.get("round", 0)) not in {1, 2}:
        raise ValueError("OPD rollout round must be 1 or 2")
    if int(row.get("rollout_index", -1)) < 0:
        raise ValueError("OPD rollout_index must be non-negative")
    for field in ("prompt_token_ids", "completion_token_ids"):
        values = row.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(value, int) and value >= 0 for value in values):
            raise ValueError(f"OPD rollout field {field!r} must be a non-empty list of token IDs")
    if row.get("prompt_token_ids_sha256") != sha256_json(row["prompt_token_ids"]):
        raise ValueError("OPD rollout prompt token hash mismatch")
    if row.get("completion_token_ids_sha256") != sha256_json(row["completion_token_ids"]):
        raise ValueError("OPD rollout completion token hash mismatch")


def summarize_rollouts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        validate_rollout_row(row)
    keys = [str(row["rollout_key"]) for row in rows]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicate_keys:
        raise ValueError(f"Duplicate OPD rollout keys; first={duplicate_keys[:3]}")
    return {
        "rows": len(rows),
        "prompts": len({str(row["record_id"]) for row in rows}),
        "rounds": dict(sorted(Counter(str(row["round"]) for row in rows).items())),
        "ordered_rollout_keys_sha256": sha256_json(keys),
        "ordered_completion_ids_sha256": sha256_json([row["completion_token_ids"] for row in rows]),
        "completion_tokens": sum(len(row["completion_token_ids"]) for row in rows),
        "empty_predictions": sum(not row["prediction"].strip() for row in rows),
    }


def _validate_score_row(row: dict[str, Any]) -> tuple[int, int]:
    if row.get("schema_version") != TEACHER_SCORE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported teacher score schema: {row.get('schema_version')!r}")
    if not isinstance(row.get("rollout_key"), str) or not row["rollout_key"]:
        raise ValueError("Teacher score rollout_key must be a non-empty string")
    completion = row.get("completion_token_ids")
    token_ids = row.get("topk_token_ids")
    logprobs = row.get("topk_logprobs")
    if not isinstance(completion, list) or not completion:
        raise ValueError("Teacher score completion_token_ids must be non-empty")
    if not isinstance(token_ids, list) or not isinstance(logprobs, list):
        raise ValueError("Teacher top-k IDs/logprobs must be lists")
    if len(token_ids) != len(completion) or len(logprobs) != len(completion):
        raise ValueError("Teacher score token dimension differs from completion length")
    widths = {len(values) for values in token_ids} | {len(values) for values in logprobs}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("Teacher score top-k rows must share one positive width")
    width = next(iter(widths))
    if not all(isinstance(value, int) and value >= 0 for values in token_ids for value in values):
        raise ValueError("Teacher top-k token IDs must be non-negative integers")
    if not all(isinstance(value, (int, float)) for values in logprobs for value in values):
        raise ValueError("Teacher top-k logprobs must be numeric")
    return len(completion), width


def write_teacher_score_shard(path: str | Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not rows:
        raise ValueError("Cannot write an empty teacher score shard")
    keys = [str(row["rollout_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Teacher score shard contains duplicate rollout keys")
    lengths_and_widths = [_validate_score_row(row) for row in rows]
    widths = {width for _, width in lengths_and_widths}
    if len(widths) != 1:
        raise ValueError("Teacher score rows must use one top-k width per shard")
    top_k = next(iter(widths))
    offsets = [0]
    for length, _ in lengths_and_widths:
        offsets.append(offsets[-1] + length)

    completion_ids = np.asarray(
        [token for row in rows for token in row["completion_token_ids"]], dtype=np.int32
    )
    topk_ids = np.asarray(
        [values for row in rows for values in row["topk_token_ids"]], dtype=np.int32
    ).reshape(-1, top_k)
    topk_logprobs = np.asarray(
        [values for row in rows for values in row["topk_logprobs"]], dtype=np.float32
    ).reshape(-1, top_k)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([TEACHER_SCORE_SCHEMA_VERSION]),
            rollout_keys=np.asarray(keys),
            prompt_token_ids_sha256=np.asarray([str(row["prompt_token_ids_sha256"]) for row in rows]),
            completion_token_ids_sha256=np.asarray(
                [str(row["completion_token_ids_sha256"]) for row in rows]
            ),
            offsets=np.asarray(offsets, dtype=np.int64),
            completion_token_ids=completion_ids,
            topk_token_ids=topk_ids,
            topk_logprobs=topk_logprobs,
        )
    temporary.replace(output)
    mass = np.exp(topk_logprobs.astype(np.float64)).sum(axis=1)
    return {
        "path": str(output.resolve()),
        "rows": len(rows),
        "tokens": int(len(completion_ids)),
        "top_k": top_k,
        "topk_mass_mean": round(float(mass.mean()), 6),
        "topk_mass_min": round(float(mass.min()), 6),
    }


def read_teacher_score_shard(path: str | Path) -> list[dict[str, Any]]:
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as payload:
        schema = str(payload["schema_version"][0])
        if schema != TEACHER_SCORE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported teacher score shard schema: {schema!r}")
        keys = payload["rollout_keys"].tolist()
        prompt_hashes = payload["prompt_token_ids_sha256"].tolist()
        completion_hashes = payload["completion_token_ids_sha256"].tolist()
        offsets = payload["offsets"].astype("int64").tolist()
        completion = payload["completion_token_ids"]
        token_ids = payload["topk_token_ids"]
        logprobs = payload["topk_logprobs"]
        if len(offsets) != len(keys) + 1 or offsets[0] != 0 or offsets[-1] != len(completion):
            raise ValueError("Teacher score shard offsets are invalid")
        rows = []
        for index, key in enumerate(keys):
            start, end = offsets[index], offsets[index + 1]
            row = {
                "schema_version": schema,
                "rollout_key": str(key),
                "prompt_token_ids_sha256": str(prompt_hashes[index]),
                "completion_token_ids_sha256": str(completion_hashes[index]),
                "completion_token_ids": completion[start:end].astype("int64").tolist(),
                "topk_token_ids": token_ids[start:end].astype("int64").tolist(),
                "topk_logprobs": logprobs[start:end].astype("float64").tolist(),
            }
            _validate_score_row(row)
            rows.append(row)
        return rows
