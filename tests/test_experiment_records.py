import hashlib
import json
import math
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ExperimentRecordTests(unittest.TestCase):
    def test_formal_sft_training_summary_matches_the_frozen_config(self):
        summary = load_json(RESULTS / "20260825_sft_training_summary.json")
        config = yaml.safe_load(
            (ROOT / summary["source_config"]).read_text(encoding="utf-8")
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["training"]["global_step"], 1200)
        self.assertEqual(summary["data"]["train_rows"], 9600)
        self.assertEqual(summary["data"]["eval_rows"], 512)
        self.assertTrue(math.isfinite(summary["training"]["train_loss"]))

        selected = summary["selected_hyperparameters"]
        training = config["training"]
        self.assertEqual(selected["epochs"], training["num_train_epochs"])
        self.assertEqual(selected["learning_rate"], training["learning_rate"])
        self.assertEqual(selected["lr_scheduler_type"], training["lr_scheduler_type"])
        self.assertEqual(selected["optimizer"], training["optim"])
        self.assertEqual(
            selected["effective_batch_size"],
            training["per_device_train_batch_size"]
            * training["gradient_accumulation_steps"],
        )
        self.assertEqual(selected["rank"], config["adapter"]["r"])
        self.assertEqual(selected["alpha"], config["adapter"]["alpha"])

        source_state = summary["source_state"]
        self.assertTrue(source_state["git_dirty"])
        self.assertFalse(source_state["commit_alone_reproduces_runtime_config"])
        self.assertIn(summary["source_config"], source_state["runtime_relevant_dirty_files"])
        config_sha256 = hashlib.sha256(
            (ROOT / summary["source_config"]).read_bytes()
        ).hexdigest()
        self.assertEqual(source_state["source_config_sha256"], config_sha256)
        self.assertEqual(
            source_state["authoritative_runtime_config"],
            summary["source_evidence"]["resolved_config"],
        )

        history = summary["teacher_forced_evaluation"]["history"]
        self.assertEqual(history[-1]["step"], summary["training"]["global_step"])
        self.assertEqual(
            history[-1]["eval_loss"],
            summary["teacher_forced_evaluation"]["eval_loss"],
        )
        self.assertEqual(
            history[-1]["eval_loss"], min(item["eval_loss"] for item in history)
        )

    def test_internal_sft_summary_is_self_consistent(self):
        summary = load_json(RESULTS / "20260826_sft_internal_summary.json")
        paired = summary["paired_base_to_sft"]

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["rows"], summary["unique_ids"])
        self.assertEqual(paired["baseline_rows"], summary["rows"])
        self.assertEqual(paired["candidate_rows"], summary["rows"])
        self.assertEqual(summary["integrity"]["manifest_identity"], "PASS")
        self.assertEqual(summary["integrity"]["metrics_recompute"], "PASS")

        zero_checks = (
            "duplicate_ids",
            "evaluation_order_mismatches",
            "reference_mismatches",
            "wrong_split_rows",
            "empty",
            "token_cap_hits",
            "missing_prompt_tokens",
            "missing_completion_tokens",
            "missing_latency",
            "invalid_latency",
        )
        for name in zero_checks:
            self.assertEqual(summary["integrity"][name], 0, name)

        for name, values in paired.items():
            if not isinstance(values, dict) or "candidate_mean" not in values:
                continue
            expected_metric = "hard_overall" if name == "hard_overall" else name
            self.assertEqual(values["candidate_mean"], summary["metrics"][expected_metric])
            self.assertAlmostEqual(
                values["candidate_mean"] - values["baseline_mean"],
                values["delta"],
                delta=0.000002,
            )
            low, high = values["bootstrap_95_ci"]
            self.assertLessEqual(low, values["delta"])
            self.assertGreaterEqual(high, values["delta"])
            self.assertGreater(low, 0)
            self.assertGreaterEqual(values["probability_delta_positive"], 0.999)

    def test_grpo_smoke_records_parameter_updates_without_capability_claims(self):
        summary = load_json(RESULTS / "20260826_grpo_smoke_summary.json")
        adapter = summary["adapter"]
        reload_summary = adapter["reload"]

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["global_step"], 2)
        self.assertTrue(math.isfinite(summary["training"]["train_loss"]))
        self.assertNotEqual(adapter["input_sha256"], adapter["output_sha256"])
        self.assertEqual(adapter["input_tensors"], adapter["output_tensors"])
        self.assertEqual(adapter["common_tensors"], adapter["changed_tensors"])
        self.assertEqual(adapter["changed_tensors"], reload_summary["trainable_tensors"])
        self.assertEqual(reload_summary["status"], "PASS")
        self.assertTrue(reload_summary["quantized_base"])
        self.assertIn("not a capability result", " ".join(summary["notes"]))

    def test_matrix_points_to_reviewed_summaries_and_keeps_r1_running(self):
        matrix = yaml.safe_load(
            (ROOT / "experiments" / "experiment_matrix.yaml").read_text(
                encoding="utf-8"
            )
        )
        experiments = {item["id"]: item for item in matrix["experiments"]}

        sft = experiments["S1_qlora_r16_domain"]
        self.assertEqual(sft["status"], "completed")
        for path in sft["evidence"].values():
            if path.startswith("experiments/results/"):
                self.assertTrue((ROOT / path).is_file(), path)

        smoke = experiments["R0_grpo_4090_smoke"]
        self.assertEqual(smoke["status"], "completed")
        self.assertTrue((ROOT / smoke["evidence"]).is_file())

        full_grpo = experiments["R1_grpo_48g_full"]
        self.assertEqual(full_grpo["status"], "running")
        self.assertNotIn("result", full_grpo)


if __name__ == "__main__":
    unittest.main()
