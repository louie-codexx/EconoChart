from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from econochart.config import load_config
from econochart.distillation.qualification import qualify_teacher
from econochart.io import write_jsonl


def _prediction(record_id: str, *, split: str, answer: str, prediction: str) -> dict:
    return {
        "id": record_id,
        "dataset": "chartqa",
        "split": split,
        "image": f"images/{record_id}.png",
        "question": f"Question {record_id}?",
        "answer": answer,
        "ground_truth": {"answer_aliases": [answer]},
        "prediction": prediction,
    }


class OpdQualificationTests(unittest.TestCase):
    def test_shipped_teacher_and_student_configs_share_frozen_inputs(self) -> None:
        student = load_config("configs/eval/opd_student_qualification.yaml")
        teacher = load_config("configs/eval/opd_teacher_qualification.yaml")

        self.assertEqual(student["data"], teacher["data"])
        self.assertEqual(student["generation"], teacher["generation"])
        self.assertEqual(student["evaluation"], teacher["evaluation"])
        self.assertEqual(student["data"]["test"][0]["expected_rows"], 256)
        self.assertEqual(student["evaluation"]["expected_split"], "val")
        self.assertEqual(student["model"]["adapter_path"], "outputs/grpo_qlora_48g_domain_v1/final_adapter")
        self.assertIsNone(teacher["model"]["adapter_path"])
        self.assertEqual(teacher["model"]["quantization"]["mode"], "none")
        self.assertNotEqual(student["training"]["output_dir"], teacher["training"]["output_dir"])

    def test_qualification_gate_writes_auditable_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            output_path = root / "report.json"
            baseline = [
                _prediction("val-1", split="val", answer="10", prediction="wrong"),
                _prediction("val-2", split="val", answer="20", prediction="20"),
            ]
            candidate = [
                _prediction("val-1", split="val", answer="10", prediction="10"),
                _prediction("val-2", split="val", answer="20", prediction="20"),
            ]
            write_jsonl(baseline_path, baseline)
            write_jsonl(candidate_path, candidate)
            config = {
                "experiment": {"id": "fixture"},
                "seed": 7,
                "bootstrap_samples": 100,
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
                    "core_metrics": ["exact_match", "relaxed_accuracy"],
                    "minimum_positive_core_metrics": 1,
                },
                "decision": {"pass_action": "smoke"},
            }

            report = qualify_teacher(config)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["gate"], "OPD_TEACHER_QUALIFICATION_PASS")
            self.assertTrue(output_path.is_file())
            self.assertEqual(report["inputs"]["baseline"]["rows"], 2)

    def test_qualification_rejects_final_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            rows = [_prediction("test-1", split="test", answer="10", prediction="10")]
            write_jsonl(baseline_path, rows)
            write_jsonl(candidate_path, rows)
            config = {
                "inputs": {
                    "baseline": str(baseline_path),
                    "candidate": str(candidate_path),
                    "expected_rows": 1,
                    "expected_split": "test",
                },
                "output": str(root / "report.json"),
                "gates": {"required": {"exact_match": {"min_delta": 0.0}}},
            }
            with self.assertRaisesRegex(ValueError, "restricted to split=val"):
                qualify_teacher(config)


if __name__ == "__main__":
    unittest.main()
