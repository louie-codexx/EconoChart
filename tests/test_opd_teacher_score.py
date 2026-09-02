from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from econochart.distillation.artifacts import (
    ROLLOUT_SCHEMA_VERSION,
    TEACHER_SCORE_SCHEMA_VERSION,
    record_identity,
    sha256_json,
    write_teacher_score_shard,
)
from econochart.distillation.teacher_score import _restored_shard_summary, align_rollouts_to_records


def _record(record_id: str, count: int = 1) -> dict:
    return {
        "id": record_id,
        "dataset": "econochart-v2.0",
        "split": "train",
        "image": f"images/{record_id}.png",
        "question": "Question?",
        "ground_truth": {"answer_aliases": ["1"]},
        "metadata": {"opd": {"round": 1, "rollout_count": count}},
    }


def _rollout(record: dict, index: int = 0) -> dict:
    prompt = [1, 2]
    completion = [3]
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollout_key": f"{record['id']}:round1:sample{index}",
        "record_id": record["id"],
        "record_sha256": record_identity(record),
        "dataset": record["dataset"],
        "split": "train",
        "image": record["image"],
        "round": 1,
        "rollout_index": index,
        "prediction": "answer",
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": sha256_json(prompt),
        "completion_token_ids": completion,
        "completion_token_ids_sha256": sha256_json(completion),
    }


class OpdTeacherScoreTests(unittest.TestCase):
    def test_rollouts_align_to_frozen_records_and_requested_counts(self) -> None:
        first = _record("train-a")
        second = _record("train-b", count=2)
        rollouts = [_rollout(first), _rollout(second), _rollout(second, index=1)]

        aligned = align_rollouts_to_records([first, second], rollouts, round_id=1)

        self.assertEqual(len(aligned), 3)
        self.assertEqual([row[0]["record_id"] for row in aligned], ["train-a", "train-b", "train-b"])

    def test_changed_record_or_missing_rollout_is_rejected(self) -> None:
        record = _record("train-a", count=2)
        with self.assertRaisesRegex(ValueError, "rollout count mismatch"):
            align_rollouts_to_records([record], [_rollout(record)], round_id=1)

        rollout = _rollout(record)
        rollout["record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity changed"):
            align_rollouts_to_records([record], [rollout, _rollout(record, index=1)], round_id=1)

    def test_restored_shard_mass_is_weighted_per_token(self) -> None:
        rows = []
        for key, token_count, mass in (("short", 1, 1.0), ("long", 3, 0.5)):
            completion = list(range(token_count))
            rows.append(
                {
                    "schema_version": TEACHER_SCORE_SCHEMA_VERSION,
                    "rollout_key": key,
                    "prompt_token_ids_sha256": "a" * 64,
                    "completion_token_ids_sha256": "b" * 64,
                    "completion_token_ids": completion,
                    "topk_token_ids": [[1] for _ in completion],
                    "topk_logprobs": [[math.log(mass)] for _ in completion],
                }
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scores.npz"
            write_teacher_score_shard(path, rows)
            summary = _restored_shard_summary(path, ["short", "long"])
            self.assertFalse((path.parent / ".scores.npz.incomplete").exists())
        self.assertEqual(summary["topk_mass_mean"], 0.625)
        self.assertEqual(summary["topk_mass_min"], 0.5)


if __name__ == "__main__":
    unittest.main()
