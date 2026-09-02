from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from econochart.config import load_config
from econochart.distillation.artifacts import (
    ROLLOUT_SCHEMA_VERSION,
    TEACHER_SCORE_SCHEMA_VERSION,
    record_identity,
    sha256_json,
    write_teacher_score_shard,
)
from econochart.distillation.train import (
    _load_scores,
    align_training_artifacts,
    validate_resume_position,
    validate_step_budget,
)
from econochart.io import sha256_file, write_json


def _record() -> dict:
    return {
        "id": "train-a",
        "dataset": "econochart-v2.0",
        "split": "train",
        "image": "images/train-a.png",
        "question": "Question?",
        "ground_truth": {"answer_aliases": ["1"]},
    }


def _artifacts() -> tuple[dict, dict, dict]:
    record = _record()
    prompt = [1, 2]
    completion = [3, 4]
    rollout = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollout_key": "train-a:round1:sample0",
        "record_id": record["id"],
        "record_sha256": record_identity(record),
        "dataset": record["dataset"],
        "split": "train",
        "image": record["image"],
        "round": 1,
        "rollout_index": 0,
        "prediction": "answer",
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": sha256_json(prompt),
        "completion_token_ids": completion,
        "completion_token_ids_sha256": sha256_json(completion),
    }
    score = {
        "schema_version": TEACHER_SCORE_SCHEMA_VERSION,
        "rollout_key": rollout["rollout_key"],
        "prompt_token_ids_sha256": rollout["prompt_token_ids_sha256"],
        "completion_token_ids_sha256": rollout["completion_token_ids_sha256"],
        "completion_token_ids": completion,
        "topk_token_ids": [[3, 1], [4, 2]],
        "topk_logprobs": [[-0.1, -3.0], [-0.2, -2.0]],
    }
    return record, rollout, score


class OpdTrainTests(unittest.TestCase):
    def test_training_artifacts_align_exactly(self) -> None:
        record, rollout, score = _artifacts()
        aligned = align_training_artifacts([record], [rollout], [score], round_id=1)
        self.assertEqual(aligned, [(record, rollout, score)])

    def test_changed_completion_or_unknown_score_is_rejected(self) -> None:
        record, rollout, score = _artifacts()
        score["completion_token_ids"] = [9, 9]
        with self.assertRaisesRegex(ValueError, "completion_token_ids"):
            align_training_artifacts([record], [rollout], [score], round_id=1)

        _, _, extra = _artifacts()
        extra["rollout_key"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown rollout"):
            align_training_artifacts([record], [rollout], [_artifacts()[2], extra], round_id=1)

    def test_round1_config_locks_cost_and_does_not_auto_start_round2(self) -> None:
        config = load_config("configs/opd/round1_student_train_v1.yaml")
        training = config["training"]
        self.assertEqual(
            config["model"]["expected_adapter_sha256"],
            "d47c083114e26000c1df5bd52676a1ca9dc2ca708a70c7944c97b3824404486b",
        )
        self.assertIn("student_model_audit", config["prerequisites"])
        self.assertEqual(training["expected_micro_steps"], 3600)
        self.assertEqual(training["gradient_accumulation_steps"], 8)
        self.assertEqual(training["expected_optimizer_steps"], 450)
        self.assertEqual(training["learning_rate"], 0.00001)
        self.assertEqual(config["distillation"]["top_k"], 16)
        self.assertFalse(config["post_training"]["automatic_round_2"])
        self.assertEqual(validate_step_budget(3600, 8, 450), 450)

    def test_non_divisible_step_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_step_budget(10, 8, 2)

    def test_resume_requires_an_accumulation_boundary(self) -> None:
        validate_resume_position(
            {"optimizer_step": 2, "next_micro_step": 16},
            accumulation_steps=8,
            total_micro_steps=32,
            total_optimizer_steps=4,
        )
        with self.assertRaisesRegex(ValueError, "accumulation boundary"):
            validate_resume_position(
                {"optimizer_step": 2, "next_micro_step": 15},
                accumulation_steps=8,
                total_micro_steps=32,
                total_optimizer_steps=4,
            )

    def test_teacher_score_shards_are_portable_between_hosts(self) -> None:
        _, _, score = _artifacts()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shard_path = root / "teacher_scores_00000.npz"
            write_teacher_score_shard(shard_path, [score])
            manifest_path = root / "manifest.json"
            write_json(
                manifest_path,
                {
                    "gate": "OPD_TEACHER_SCORE_ARTIFACT_PASS",
                    "status": "passed",
                    "shards": [
                        {
                            "path": "/teacher-host/teacher_scores_00000.npz",
                            "sha256": sha256_file(shard_path),
                        }
                    ],
                },
            )
            resolved_manifest, rows, _ = _load_scores(
                {"manifest": str(manifest_path), "expected_rows": 1}
            )
            self.assertEqual(resolved_manifest, manifest_path.resolve())
            self.assertEqual(rows[0]["rollout_key"], score["rollout_key"])


if __name__ == "__main__":
    unittest.main()
