from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from econochart.evaluation.runner import _validate_resume_manifest, run
from econochart.io import write_json, write_jsonl


class EvaluationResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "seed": 20260821,
            "data": {"test": [{"path": "frozen_test.jsonl"}]},
            "generation": {"do_sample": False, "max_new_tokens": 320},
            "model": {
                "name_or_path": "frozen-base",
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "min_pixels": 200704,
                "max_pixels": 802816,
                "quantization": {"mode": "4bit", "compute_dtype": "bfloat16"},
            },
        }

    def _write_manifest(self, output_dir: Path) -> None:
        write_json(
            output_dir / "run_manifest.json",
            {
                "model": {"base": "frozen-base", "adapter": "frozen-adapter"},
                "config": self.config,
            },
        )

    def test_matching_model_settings_allow_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self._write_manifest(output_dir)
            _validate_resume_manifest(
                output_dir,
                config=copy.deepcopy(self.config),
                model_path="frozen-base",
                adapter_path="frozen-adapter",
            )

    def test_model_setting_drift_is_rejected_before_gpu_loading(self) -> None:
        changes = {
            "dtype": "float16",
            "attn_implementation": "eager",
            "min_pixels": 100352,
            "max_pixels": 401408,
            "quantization": {"mode": "none", "compute_dtype": "bfloat16"},
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self._write_manifest(output_dir)
            prediction_path = output_dir / "predictions.jsonl"
            write_jsonl(prediction_path, [{"id": "sample-1", "prediction": "original"}])
            original_predictions = prediction_path.read_bytes()
            with (
                patch("econochart.evaluation.runner._load_eval_records", return_value=[{"id": "sample-1"}]),
                patch("econochart.evaluation.runner.load_base_model") as load_model,
            ):
                for key, value in changes.items():
                    with self.subTest(setting=key):
                        changed = copy.deepcopy(self.config)
                        changed["model"][key] = value
                        with self.assertRaisesRegex(ValueError, "Resume config field 'model' differs"):
                            run(
                                changed,
                                model_override="frozen-base",
                                adapter_override="frozen-adapter",
                                output_override=str(output_dir),
                                resume=True,
                            )
                        load_model.assert_not_called()
                        self.assertEqual(prediction_path.read_bytes(), original_predictions)


if __name__ == "__main__":
    unittest.main()
