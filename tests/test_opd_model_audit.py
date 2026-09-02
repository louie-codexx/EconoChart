from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from econochart.distillation.model_audit import audit_model_snapshot


class OpdModelAuditTests(unittest.TestCase):
    def _snapshot(self, root: Path, *, include_second_shard: bool = True) -> None:
        (root / "config.json").write_text(
            json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}),
            encoding="utf-8",
        )
        (root / "tokenizer.json").write_text('{"model":"fixture"}\n', encoding="utf-8")
        (root / "preprocessor_config.json").write_text('{"image":"fixture"}\n', encoding="utf-8")
        weight_map = {"layer.0": "model-00001-of-00002.safetensors", "layer.1": "model-00002-of-00002.safetensors"}
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        (root / "model-00001-of-00002.safetensors").write_bytes(b"first")
        if include_second_shard:
            (root / "model-00002-of-00002.safetensors").write_bytes(b"second")

    def test_complete_snapshot_passes_without_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._snapshot(root)

            report = audit_model_snapshot(root, expected_shards=2, expected_model_type="qwen3_vl")

            self.assertEqual(report["status"], "passed", report["issues"])
            self.assertEqual(report["weights"]["referenced_shards"], 2)
            self.assertIsNotNone(report["tokenizer_processor"]["fingerprint_sha256"])
            self.assertFalse(report["weights"]["headers_validated"])

    def test_missing_shard_and_partial_file_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._snapshot(root, include_second_shard=False)
            (root / "download.incomplete").write_bytes(b"partial")

            report = audit_model_snapshot(root, expected_shards=2, expected_model_type="qwen3_vl")

            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("missing referenced shard" in issue for issue in report["issues"]))
            self.assertTrue(any("temporary files remain" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
