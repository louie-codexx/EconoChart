from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from econochart.distillation.artifacts import (
    ROLLOUT_SCHEMA_VERSION,
    TEACHER_SCORE_SCHEMA_VERSION,
    read_teacher_score_shard,
    sha256_json,
    summarize_rollouts,
    validate_adapter_snapshot,
    validate_resume_snapshot,
    write_teacher_score_shard,
)
from econochart.io import write_json


def _rollout(key: str, completion: list[int]) -> dict:
    prompt = [1, 2, 3]
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollout_key": key,
        "record_id": key.split(":")[0],
        "record_sha256": "a" * 64,
        "dataset": "econochart-v2.0",
        "split": "train",
        "image": "images/train/chart.png",
        "round": 1,
        "rollout_index": 0,
        "prediction": "answer",
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": sha256_json(prompt),
        "completion_token_ids": completion,
        "completion_token_ids_sha256": sha256_json(completion),
    }


def _score(row: dict, top_k: int = 2) -> dict:
    return {
        "schema_version": TEACHER_SCORE_SCHEMA_VERSION,
        "rollout_key": row["rollout_key"],
        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        "completion_token_ids_sha256": row["completion_token_ids_sha256"],
        "completion_token_ids": row["completion_token_ids"],
        "topk_token_ids": [[index, index + 1][:top_k] for index in row["completion_token_ids"]],
        "topk_logprobs": [[-0.2, -2.0][:top_k] for _ in row["completion_token_ids"]],
    }


class OpdArtifactTests(unittest.TestCase):
    def test_adapter_snapshot_requires_the_frozen_weight_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(b"adapter")
            (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            expected = hashlib.sha256(b"adapter").hexdigest()
            identity = validate_adapter_snapshot(root, expected)
            self.assertEqual(identity["weights_sha256"], expected)
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                validate_adapter_snapshot(root, "0" * 64)

    def test_rollout_summary_locks_order_and_token_ids(self) -> None:
        rows = [_rollout("a:0", [4, 5]), _rollout("b:0", [6])]
        summary = summarize_rollouts(rows)
        repeated = summarize_rollouts(rows)

        self.assertEqual(summary, repeated)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["prompts"], 2)
        self.assertEqual(summary["completion_tokens"], 3)

    def test_teacher_score_npz_round_trip_preserves_variable_lengths(self) -> None:
        rollouts = [_rollout("a:0", [4, 5]), _rollout("b:0", [6])]
        scores = [_score(row) for row in rollouts]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scores.npz"
            summary = write_teacher_score_shard(path, scores)
            restored = read_teacher_score_shard(path)

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["tokens"], 3)
        self.assertEqual(summary["top_k"], 2)
        self.assertEqual([row["completion_token_ids"] for row in restored], [[4, 5], [6]])
        self.assertEqual([row["rollout_key"] for row in restored], ["a:0", "b:0"])

    def test_mismatched_teacher_score_shape_is_rejected(self) -> None:
        rollout = _rollout("a:0", [4, 5])
        score = _score(rollout)
        score["topk_logprobs"] = score["topk_logprobs"][:-1]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "completion length"):
                write_teacher_score_shard(Path(temp_dir) / "scores.npz", [score])

    def test_resume_snapshot_binds_stage_model_adapter_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_json(
                output / "run_manifest.json",
                {
                    "stage": "opd_student_rollout",
                    "model": {"base": "base", "adapter": "adapter"},
                    "data": {"source_sha256": "a" * 64},
                    "config": {"seed": 7},
                },
            )
            snapshot = validate_resume_snapshot(
                output,
                stage="opd_student_rollout",
                config={"seed": 7, "_config_path": "ignored"},
                model_path="base",
                adapter_path="adapter",
                data_summary={"source_sha256": "a" * 64},
            )
            self.assertEqual(snapshot["stage"], "opd_student_rollout")
            with self.assertRaisesRegex(ValueError, "config differs"):
                validate_resume_snapshot(
                    output,
                    stage="opd_student_rollout",
                    config={"seed": 8},
                    model_path="base",
                    adapter_path="adapter",
                )
            with self.assertRaisesRegex(ValueError, "inputs or prerequisite"):
                validate_resume_snapshot(
                    output,
                    stage="opd_student_rollout",
                    config={"seed": 7},
                    model_path="base",
                    adapter_path="adapter",
                    data_summary={"source_sha256": "b" * 64},
                )


class OpdLossTests(unittest.TestCase):
    def test_sparse_forward_kl_is_zero_for_matching_distribution_and_has_finite_gradients(self) -> None:
        import torch

        from econochart.distillation.losses import sparse_forward_kl_loss

        logits = torch.tensor([[[2.0, 1.0, 0.0]]], requires_grad=True)
        teacher_logprobs = torch.log_softmax(logits.detach(), dim=-1)
        top_values, top_ids = teacher_logprobs.topk(2, dim=-1)
        loss = sparse_forward_kl_loss(logits, top_ids, top_values)
        loss.backward()

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_sparse_forward_kl_increases_for_shifted_student(self) -> None:
        import torch

        from econochart.distillation.losses import sparse_forward_kl_loss

        teacher_logits = torch.tensor([[[3.0, 1.0, 0.0]]])
        teacher_logprobs = torch.log_softmax(teacher_logits, dim=-1)
        top_values, top_ids = teacher_logprobs.topk(2, dim=-1)
        student_logits = torch.tensor([[[0.0, 1.0, 3.0]]], requires_grad=True)

        loss = sparse_forward_kl_loss(student_logits, top_ids, top_values)

        self.assertGreater(float(loss.item()), 0.1)


if __name__ == "__main__":
    unittest.main()
