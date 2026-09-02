from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from econochart.config import load_config
from econochart.distillation.promotion import qualify_round
from econochart.io import write_jsonl


def _prediction(record_id: str, *, answer: str, prediction: str) -> dict:
    return {
        "id": record_id,
        "dataset": "chartqa",
        "split": "val",
        "image": f"images/{record_id}.png",
        "question": f"Question {record_id}?",
        "answer": answer,
        "ground_truth": {"answer_aliases": [answer]},
        "prediction": prediction,
    }


class OpdPromotionTests(unittest.TestCase):
    def test_development_configs_use_same_frozen_inputs_and_different_adapters(self) -> None:
        baseline = load_config("configs/eval/opd_round1_baseline_dev.yaml")
        candidate = load_config("configs/eval/opd_round1_candidate_dev.yaml")
        gate = load_config("configs/opd/round1_development_gate_v1.yaml")

        self.assertEqual(baseline["data"], candidate["data"])
        self.assertEqual(baseline["generation"], candidate["generation"])
        self.assertEqual(baseline["evaluation"], candidate["evaluation"])
        self.assertEqual(baseline["evaluation"]["expected_split"], "val")
        self.assertEqual(baseline["data"]["test"][0]["expected_rows"], 256)
        self.assertNotEqual(baseline["model"]["adapter_path"], candidate["model"]["adapter_path"])
        self.assertFalse(gate["decision"]["automatic_round_2"])
        self.assertEqual(gate["gates"]["required"]["numeric_recall"]["min_delta"], 0.05)

    def test_round_gate_uses_distinct_val_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            output_path = root / "gate.json"
            write_jsonl(
                baseline_path,
                [
                    _prediction("val-1", answer="10", prediction="wrong"),
                    _prediction("val-2", answer="20", prediction="20"),
                ],
            )
            write_jsonl(
                candidate_path,
                [
                    _prediction("val-1", answer="10", prediction="10"),
                    _prediction("val-2", answer="20", prediction="20"),
                ],
            )
            report = qualify_round(
                {
                    "inputs": {
                        "baseline": str(baseline_path),
                        "candidate": str(candidate_path),
                        "expected_rows": 2,
                        "expected_split": "val",
                    },
                    "output": str(output_path),
                    "gates": {
                        "required": {"exact_match": {"min_delta": 0.4}},
                        "point_estimate_guardrails": {"relaxed_accuracy": 0.0},
                        "core_metrics": ["exact_match"],
                        "minimum_positive_core_metrics": 1,
                    },
                }
            )
            self.assertEqual(report["gate"], "OPD_ROUND1_DEVELOPMENT_GATE_PASS")
            self.assertTrue(output_path.is_file())
