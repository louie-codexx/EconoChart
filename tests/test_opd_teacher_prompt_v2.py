from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from econochart.config import load_config
from econochart.data.schema import parse_ground_truth
from econochart.data.training import (
    DOMAIN_SYSTEM_PROMPT,
    OPD_TEACHER_PROMPT_PROFILE_V2,
    load_record_sources,
    opd_teacher_required_sections,
    system_prompt_for,
)
from econochart.distillation.prompt_smoke import evaluate_prompt_smoke
from econochart.distillation.qualification import _validate_teacher_candidate_contract
from econochart.evaluation.inference import _build_messages
from econochart.evaluation.runner import _load_eval_records, _validate_prerequisite_gate
from econochart.io import read_jsonl, write_json, write_jsonl
from econochart.preflight import run_preflight

TASKS = (
    "comprehensive_report",
    "decision_support",
    "numerical_reasoning",
    "relationship_analysis",
    "risk_diagnosis",
    "trend_analysis",
    "value_retrieval",
)


def _record(record_id: str, task_type: str, *, split: str = "train") -> dict:
    sections = list(opd_teacher_required_sections(task_type))
    return {
        "id": record_id,
        "dataset": "econochart-v2.0",
        "split": split,
        "chart_id": f"chart-{record_id}",
        "entity_id": f"entity-{record_id}",
        "image": f"images/{record_id}.png",
        "question": "请只依据图表回答这个问题。",
        "answer": "POISONED_REFERENCE_ANSWER",
        "ground_truth": {
            "required_sections": sections,
            "length_range": [1, 1000],
        },
        "task_type": task_type,
        "view_type": "dashboard",
        "industry": "finance",
        "difficulty": "hard",
    }


def _prediction(record_id: str, task_type: str, *, split: str = "train") -> dict:
    row = _record(record_id, task_type, split=split)
    row["prediction"] = "\n".join(
        f"【{section}】合规内容" for section in opd_teacher_required_sections(task_type)
    )
    row["latency_seconds"] = 1.0
    row["completion_tokens"] = 20
    return row


class OpdTeacherPromptV2Tests(unittest.TestCase):
    def test_prompt_profile_is_explicit_strict_and_label_blind(self) -> None:
        record = _record("decision", "decision_support")
        record["ground_truth"]["numeric_targets"] = [{"value": 999999}]

        prompt = system_prompt_for(record, OPD_TEACHER_PROMPT_PROFILE_V2)

        self.assertEqual(system_prompt_for(record), DOMAIN_SYSTEM_PROMPT)
        self.assertNotIn("POISONED_REFERENCE_ANSWER", prompt)
        self.assertNotIn("999999", prompt)
        self.assertIn("【结论】、【数据依据】、【风险】、【建议】", prompt)
        self.assertIn("不得改成Markdown标题", prompt)
        self.assertIn("不要用整段原始年度序列代替被问目标", prompt)

        value_prompt = system_prompt_for(
            _record("value", "value_retrieval"),
            OPD_TEACHER_PROMPT_PROFILE_V2,
        )
        self.assertNotIn("【风险】使用", value_prompt)
        self.assertNotIn("【建议】必须", value_prompt)

    def test_prompt_profile_maps_sections_from_task_and_view_without_labels(self) -> None:
        expected = {
            "value_retrieval": ("结论", "数据依据"),
            "numerical_reasoning": ("结论", "数据依据"),
            "relationship_analysis": ("结论", "数据依据"),
            "trend_analysis": ("结论", "数据依据", "风险"),
            "risk_diagnosis": ("结论", "数据依据", "风险"),
            "decision_support": ("结论", "数据依据", "风险", "建议"),
            "comprehensive_report": ("结论", "数据依据", "风险", "建议"),
        }
        for task_type, sections in expected.items():
            self.assertEqual(opd_teacher_required_sections(task_type), sections)
        self.assertEqual(
            opd_teacher_required_sections("relationship_analysis", view_type="financial"),
            ("结论", "数据依据", "风险"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported OPD teacher task type"):
            opd_teacher_required_sections("unknown")

    def test_label_blind_section_policy_matches_generated_domain_contract(self) -> None:
        path = Path("data/samples/econochart_v2/annotations/train.jsonl")
        for row in read_jsonl(path):
            expected = tuple(parse_ground_truth(row["ground_truth"])["required_sections"])
            observed = opd_teacher_required_sections(
                str(row["task_type"]),
                view_type=str(row["view_type"]),
            )
            self.assertEqual(observed, expected, row["id"])

    def test_v2_profile_is_rejected_for_public_evaluation_records(self) -> None:
        record = _record("public", "value_retrieval")
        record["dataset"] = "chartqa"
        with self.assertRaisesRegex(ValueError, "restricted to EconoChart"):
            system_prompt_for(record, OPD_TEACHER_PROMPT_PROFILE_V2)

    def test_inference_message_uses_requested_prompt_profile(self) -> None:
        messages = _build_messages(
            _record("risk", "risk_diagnosis"),
            Image.new("RGB", (2, 2)),
            prompt_profile=OPD_TEACHER_PROMPT_PROFILE_V2,
        )
        system_text = messages[0]["content"][0]["text"]
        self.assertIn("【风险】", system_text)
        self.assertIn("只选最重要的两项风险", system_text)

    def test_task_quota_sampling_is_exact_deterministic_and_train_only(self) -> None:
        rows = [
            _record(f"{task}-{index}", task)
            for task in TASKS
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            write_jsonl(path, rows)
            source = {
                "path": str(path),
                "sampling_namespace": "teacher-prompt-v2-fixture",
                "task_quotas": {task: 2 for task in TASKS},
                "expected_rows": 14,
            }
            first = load_record_sources([source], expected_split="train", seed=20260821)
            second = load_record_sources([source], expected_split="train", seed=20260821)

            self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
            self.assertEqual(Counter(row["task_type"] for row in first), Counter({task: 2 for task in TASKS}))
            self.assertTrue(all(row["split"] == "train" for row in first))

            invalid = {**source, "max_samples": 14}
            with self.assertRaisesRegex(ValueError, "both max_samples and task_quotas"):
                load_record_sources([invalid], expected_split="train", seed=20260821)

    def test_evaluation_loader_accepts_explicit_training_calibration(self) -> None:
        rows = [_record(f"value-{index}", "value_retrieval") for index in range(2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            write_jsonl(path, rows)
            loaded = _load_eval_records(
                {
                    "seed": 7,
                    "data": {"test": [{"path": str(path), "expected_rows": 2}]},
                    "evaluation": {"expected_split": "train"},
                }
            )
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(row["split"] == "train" for row in loaded))

    def test_inputs_only_preflight_reports_task_quota_effective_rows(self) -> None:
        config = {
            "seed": 20260821,
            "data": {
                "test": [
                    {
                        "path": "data/samples/econochart_v2/annotations/train.jsonl",
                        "sampling_namespace": "teacher-prompt-v2-preflight-fixture",
                        "task_quotas": {task: 1 for task in TASKS},
                        "expected_rows": 7,
                    }
                ]
            },
            "evaluation": {"expected_split": "train"},
        }
        report = run_preflight(config, stage="eval", inputs_only=True)
        source = report["checks"]["data"]["test"]["sources"][0]
        selected = report["checks"]["data"]["test"]["selected"]
        self.assertEqual(report["status"], "passed", report["issues"])
        self.assertEqual(source["effective_rows"], 7)
        self.assertEqual(selected["rows"], 7)

    def test_smoke_gate_requires_train_task_balance_and_exact_protocol(self) -> None:
        rows = [_prediction(f"row-{index}", task) for index, task in enumerate(TASKS)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "predictions.jsonl"
            write_jsonl(predictions, rows)
            config = {
                "inputs": {
                    "predictions": str(predictions),
                    "expected_rows": 7,
                    "expected_task_counts": {task: 1 for task in TASKS},
                },
                "output": str(root / "gate.json"),
                "gates": {
                    "minimum_overall": {"overall": 0.0, "format": 1.0},
                    "minimum_task_metrics": {task: {"overall": 0.0} for task in TASKS},
                    "maximum_efficiency": {"mean_completion_tokens": 30},
                    "maximum_protocol_failures": 0,
                },
            }

            report = evaluate_prompt_smoke(config)
            self.assertEqual(report["gate"], "OPD_TEACHER_PROMPT_SMOKE_PASS")

            rows[0]["prediction"] = f"**一、结论**\n{rows[0]['prediction']}"
            write_jsonl(predictions, rows)
            failed_config = {**config, "output": str(root / "failed-gate.json")}
            failed = evaluate_prompt_smoke(failed_config)
            self.assertEqual(failed["gate"], "OPD_TEACHER_PROMPT_SMOKE_FAIL")
            self.assertEqual(failed["gate_evaluation"]["checks"][-1]["observed"], 1)

    def test_v2_evaluation_and_qualification_require_smoke_pass_and_profile_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_path = root / "smoke-gate.json"
            manifest_path = root / "run_manifest.json"
            evaluation_config = {
                "experiment": {
                    "prerequisite_gate": str(gate_path),
                    "prerequisite_gate_name": "OPD_TEACHER_PROMPT_SMOKE_PASS",
                }
            }
            qualification_config = {
                "inputs": {
                    "prerequisite_gate": str(gate_path),
                    "prerequisite_gate_name": "OPD_TEACHER_PROMPT_SMOKE_PASS",
                    "candidate_manifest": str(manifest_path),
                    "required_prompt_profile": OPD_TEACHER_PROMPT_PROFILE_V2,
                }
            }

            write_json(gate_path, {"status": "failed", "gate": "OPD_TEACHER_PROMPT_SMOKE_FAIL"})
            with self.assertRaisesRegex(ValueError, "prerequisite gate is not satisfied"):
                _validate_prerequisite_gate(evaluation_config)

            write_json(gate_path, {"status": "passed", "gate": "OPD_TEACHER_PROMPT_SMOKE_PASS"})
            _validate_prerequisite_gate(evaluation_config)
            write_json(manifest_path, {"config": {"generation": {"prompt_profile": "wrong"}}})
            with self.assertRaisesRegex(ValueError, "wrong prompt profile"):
                _validate_teacher_candidate_contract(qualification_config)

            write_json(
                manifest_path,
                {"config": {"generation": {"prompt_profile": OPD_TEACHER_PROMPT_PROFILE_V2}}},
            )
            _validate_teacher_candidate_contract(qualification_config)

    def test_shipped_v2_configs_preserve_v1_qualification_thresholds(self) -> None:
        smoke_eval = load_config("configs/eval/opd_teacher_prompt_smoke_v2.yaml")
        v1_eval = load_config("configs/eval/opd_teacher_qualification.yaml")
        v2_eval = load_config("configs/eval/opd_teacher_qualification_v2.yaml")
        smoke_gate = load_config("configs/opd/teacher_prompt_smoke_v2.yaml")
        v1_gate = load_config("configs/opd/teacher_qualification_v1.yaml")
        v2_gate = load_config("configs/opd/teacher_qualification_v2.yaml")

        self.assertEqual(smoke_eval["evaluation"]["expected_split"], "train")
        self.assertEqual(smoke_eval["data"]["test"][0]["expected_rows"], 28)
        self.assertEqual(sum(smoke_eval["data"]["test"][0]["task_quotas"].values()), 28)
        self.assertEqual(smoke_eval["generation"]["prompt_profile"], OPD_TEACHER_PROMPT_PROFILE_V2)
        self.assertEqual(v2_eval["generation"]["prompt_profile"], OPD_TEACHER_PROMPT_PROFILE_V2)
        self.assertEqual(v2_eval["data"], v1_eval["data"])
        self.assertNotEqual(v2_eval["training"]["output_dir"], v1_eval["training"]["output_dir"])
        self.assertEqual(v2_gate["gates"], v1_gate["gates"])
        self.assertNotEqual(v2_gate["inputs"]["candidate"], v1_gate["inputs"]["candidate"])
        self.assertEqual(smoke_gate["gates"]["minimum_overall"]["format"], 1.0)
        self.assertTrue(smoke_gate["decision"]["no_opd_rollout_is_authorized_by_this_smoke_gate"])


if __name__ == "__main__":
    unittest.main()
