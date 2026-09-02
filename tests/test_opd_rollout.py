from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from econochart.config import load_config
from econochart.distillation.artifacts import ROLLOUT_SCHEMA_VERSION, record_identity, sha256_json
from econochart.distillation.rollout import (
    _validate_resume,
    _write_rollout_checkpoint,
    expand_rollout_jobs,
    validate_image_token_presence,
)
from econochart.io import read_jsonl


def _record(record_id: str, rollout_count: int, *, round_id: int = 1) -> dict:
    return {
        "id": record_id,
        "dataset": "econochart-v2.0",
        "split": "train",
        "image": f"images/{record_id}.png",
        "question": "Question?",
        "ground_truth": {"answer_aliases": ["1"]},
        "metadata": {"opd": {"round": round_id, "rollout_count": rollout_count}},
    }


def _rollout(job: dict) -> dict:
    record = job["record"]
    prompt = [1, 9, 2]
    completion = [3]
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollout_key": job["rollout_key"],
        "record_id": record["id"],
        "record_sha256": record_identity(record),
        "dataset": record["dataset"],
        "split": record["split"],
        "image": record["image"],
        "round": record["metadata"]["opd"]["round"],
        "rollout_index": job["rollout_index"],
        "rollout_seed": job["rollout_seed"],
        "prediction": "answer",
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": sha256_json(prompt),
        "completion_token_ids": completion,
        "completion_token_ids_sha256": sha256_json(completion),
    }


class OpdRolloutTests(unittest.TestCase):
    def test_round1_config_freezes_the_student_adapter_and_model_audit(self) -> None:
        config = load_config("configs/opd/round1_rollout_v1.yaml")
        self.assertEqual(
            config["model"]["expected_adapter_sha256"],
            "d47c083114e26000c1df5bd52676a1ca9dc2ca708a70c7944c97b3824404486b",
        )
        self.assertEqual(
            config["prerequisites"]["student_model_audit"]["expected_gate"],
            "OPD_MODEL_SNAPSHOT_AUDIT_PASS",
        )

    def test_jobs_expand_one_or_two_rollouts_deterministically(self) -> None:
        rows = [_record("train-a", 1), _record("train-b", 2)]
        jobs = expand_rollout_jobs(rows, round_id=1, seed=20260821)
        repeated = expand_rollout_jobs(rows, round_id=1, seed=20260821)

        self.assertEqual(jobs, repeated)
        self.assertEqual([job["rollout_key"] for job in jobs], [
            "train-a:round1:sample0",
            "train-b:round1:sample0",
            "train-b:round1:sample1",
        ])
        self.assertEqual(len({job["rollout_seed"] for job in jobs}), 3)

    def test_wrong_round_or_rollout_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not belong"):
            expand_rollout_jobs([_record("train-a", 1, round_id=2)], round_id=1, seed=1)
        with self.assertRaisesRegex(ValueError, "one or two"):
            expand_rollout_jobs([_record("train-a", 3)], round_id=1, seed=1)

    def test_image_tokens_are_a_hard_gate(self) -> None:
        self.assertEqual(validate_image_token_presence([1, 9, 9, 2], 9), 2)
        with self.assertRaisesRegex(ValueError, "no image tokens"):
            validate_image_token_presence([1, 2, 3], 9)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_image_token_presence([1, 2, 3], None)

    def test_resume_rejects_changed_frozen_identity(self) -> None:
        jobs = expand_rollout_jobs([_record("train-a", 1)], round_id=1, seed=7)
        existing = [_rollout(jobs[0])]
        _validate_resume(existing, jobs)
        existing[0]["record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity differs"):
            _validate_resume(existing, jobs)

    def test_rollout_checkpoint_replaces_the_file_without_partial_artifact(self) -> None:
        jobs = expand_rollout_jobs([_record("train-a", 1)], round_id=1, seed=7)
        rows = [_rollout(jobs[0])]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            path.write_text("stale\n", encoding="utf-8")
            _write_rollout_checkpoint(path, rows)
            restored = list(read_jsonl(path))
            self.assertFalse((path.parent / ".rollouts.jsonl.incomplete").exists())
        self.assertEqual(restored, rows)


if __name__ == "__main__":
    unittest.main()
