from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from econochart.distillation.compatibility import compare_model_interfaces


class OpdCompatibilityTests(unittest.TestCase):
    def _model(self, root: Path, *, vocab_size: int = 3, patch_size: int = 14) -> None:
        root.mkdir()
        (root / "tokenizer.json").write_text(
            json.dumps(
                {
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<image>": 2}, "merges": []},
                    "added_tokens": [{"id": 2, "content": "<image>", "special": True}],
                    "normalizer": None,
                    "pre_tokenizer": {"type": "ByteLevel"},
                    "post_processor": None,
                    "decoder": {"type": "ByteLevel"},
                }
            ),
            encoding="utf-8",
        )
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_vl",
                    "vocab_size": vocab_size,
                    "image_token_id": 2,
                    "video_token_id": 3,
                    "vision_start_token_id": 4,
                    "vision_end_token_id": 5,
                }
            ),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "bos_token": None,
                    "eos_token": "<eos>",
                    "pad_token": "<pad>",
                    "unk_token": None,
                    "chat_template": "{{ messages }}",
                    "added_tokens_decoder": {"2": {"content": "<image>", "special": True}},
                }
            ),
            encoding="utf-8",
        )
        (root / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "patch_size": patch_size,
                    "temporal_patch_size": 2,
                    "merge_size": 2,
                    "min_pixels": None,
                    "max_pixels": None,
                    "image_mean": [0.5],
                    "image_std": [0.5],
                }
            ),
            encoding="utf-8",
        )

    def test_same_family_interfaces_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            student = root / "student"
            teacher = root / "teacher"
            self._model(student)
            self._model(teacher)

            report = compare_model_interfaces(student, teacher)

            self.assertEqual(report["status"], "passed", report["issues"])
            self.assertTrue(report["tokenizer"]["matched"])
            self.assertTrue(report["model_tokens"]["matched"])
            self.assertTrue(report["processor_config"]["matched"])

    def test_vocab_and_processor_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            student = root / "student"
            teacher = root / "teacher"
            self._model(student)
            self._model(teacher, vocab_size=4, patch_size=16)

            report = compare_model_interfaces(student, teacher)

            self.assertEqual(report["status"], "failed")
            self.assertIn("vocab_size", report["model_tokens"]["mismatched_fields"])
            self.assertIn("patch_size", report["processor_config"]["mismatched_fields"])


if __name__ == "__main__":
    unittest.main()
