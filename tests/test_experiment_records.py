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
    def test_frozen_subset_summary_records_the_audited_budgets(self):
        summary = load_json(RESULTS / "20260822_data_subset_summary.json")
        subsets = summary["subsets"]

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["seed"], 20260821)
        self.assertEqual(subsets["sft_screen"]["rows"], 4800)
        self.assertEqual(subsets["sft_final"]["rows"], 9600)
        self.assertEqual(subsets["grpo_final"]["rows"], 3600)
        self.assertEqual(subsets["development_val"]["rows"], 512)
        self.assertEqual(sum(subsets["grpo_final"]["task_quotas"].values()), 3600)
        self.assertTrue(summary["integrity"]["sft_screen_nested_in_sft_final"])
        self.assertEqual(summary["integrity"]["complete_internal_test_rows_preserved"], 2496)

    def test_base_summary_matches_later_paired_controls(self):
        base = load_json(RESULTS / "20260822_base_internal_summary.json")
        sft = load_json(RESULTS / "20260826_sft_internal_summary.json")
        grpo = load_json(RESULTS / "20260828_grpo_internal_summary.json")

        self.assertEqual(base["status"], "passed")
        self.assertEqual(base["rows"], 2496)
        self.assertEqual(base["metrics"]["format"], 0.0)
        self.assertEqual(base["metric_correction"]["predictions_regenerated"], False)
        self.assertEqual(base["hashes"]["test_sha256"], sft["hashes"]["test_sha256"])
        for name, value in base["metrics"].items():
            self.assertEqual(value, sft["paired_base_to_sft"][name]["baseline_mean"])
            self.assertEqual(value, grpo["paired_base_to_grpo"][name]["baseline_mean"])

    def test_sft_smoke_is_closed_without_a_capability_claim(self):
        summary = load_json(RESULTS / "20260822_sft_smoke_summary.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["training"]["global_step"], 2)
        self.assertTrue(summary["training"]["all_losses_finite"])
        self.assertEqual(summary["adapter"]["reload_status"], "PASS")
        self.assertEqual(summary["adapter"]["trainable_tensors"], 504)
        self.assertEqual(summary["memory"]["trainer_peak_allocated_delta_gib"], 1.813)
        self.assertIn("whole-device peak", summary["memory"]["measurement_limit"])
        self.assertIn("not a capability result", " ".join(summary["notes"]))

    def test_sft_screening_summary_locks_the_selected_parameters(self):
        summary = load_json(RESULTS / "20260825_sft_screening_summary.json")
        decision = summary["decision"]
        learning_rate = summary["paired_1e4_to_2e4"]
        rank = summary["paired_rank8_minus_rank16"]
        learning_rate_candidates = summary["learning_rate_candidates_at_rank16"]
        rank_candidates = summary["rank_candidates_at_2e4"]

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["data"]["formal_test_used_for_selection"])
        self.assertEqual(decision["selected_learning_rate"], 0.0002)
        self.assertEqual(decision["selected_rank"], 16)
        self.assertEqual(decision["selected_alpha"], 32)
        self.assertFalse(decision["rank32_triggered"])
        for name in ("overall", "numeric", "numeric_recall", "numeric_precision", "trend", "hard_overall"):
            self.assertGreater(learning_rate[name]["bootstrap_95_ci"][0], 0, name)
        for name in ("overall", "numeric", "numeric_recall", "numeric_precision", "trend"):
            self.assertLess(rank[name]["bootstrap_95_ci"][1], 0, name)

        for name, values in learning_rate.items():
            expected = learning_rate_candidates["2e-4"][name] - learning_rate_candidates["1e-4"][name]
            self.assertAlmostEqual(values["delta"], expected, delta=0.000002, msg=name)

        for name, values in rank.items():
            expected = rank_candidates["rank8"][name] - rank_candidates["rank16"][name]
            self.assertAlmostEqual(values["delta"], expected, delta=0.000002, msg=name)

        inference_speed_delta_percent = 100 * (
            rank_candidates["rank8"]["samples_per_second"]
            / rank_candidates["rank16"]["samples_per_second"]
            - 1
        )
        self.assertAlmostEqual(inference_speed_delta_percent, 1.36, delta=0.01)
        self.assertEqual(summary["execution_anomalies"][0]["excluded_rows"], 50)
        self.assertFalse(summary["execution_anomalies"][0]["excluded_rows_used_for_selection"])
        self.assertTrue(summary["source_evidence"]["rank8_to_rank16_paired_report"].endswith(".json"))

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
        self.assertEqual(summary["memory"]["trainer_peak_allocated_delta_mib"], 2790)
        self.assertIn("whole-device peak", summary["memory"]["measurement_limit"])

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

    def test_full_grpo_summary_preserves_attribution_boundary(self):
        summary = load_json(RESULTS / "20260828_grpo_internal_summary.json")
        sft_paired = summary["paired_sft_to_grpo"]
        base_paired = summary["paired_base_to_grpo"]

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["training"]["global_step"], 3600)
        self.assertEqual(summary["evaluation"]["rows"], 2496)
        self.assertEqual(summary["evaluation"]["artifact_audit"], "PASS")
        self.assertEqual(summary["adapter"]["reload_status"], "PASS")
        self.assertEqual(summary["memory_observation"]["nvidia_smi_process_mib"], 7614)
        self.assertFalse(summary["memory_observation"]["is_peak"])

        for name, values in sft_paired.items():
            if not isinstance(values, dict) or "candidate_mean" not in values:
                continue
            self.assertEqual(values["candidate_mean"], summary["metrics"][name])
            low, high = values["bootstrap_95_ci"]
            self.assertLessEqual(low, 0, name)
            self.assertGreaterEqual(high, 0, name)
            self.assertEqual(values["verdict"], "INCONCLUSIVE")

        for name, values in base_paired.items():
            if not isinstance(values, dict) or "candidate_mean" not in values:
                continue
            self.assertEqual(values["candidate_mean"], summary["metrics"][name])
            self.assertGreater(values["bootstrap_95_ci"][0], 0, name)
            self.assertEqual(values["verdict"], "GRPO_BETTER")

    def test_matrix_points_to_reviewed_summaries_and_closes_r1(self):
        matrix = yaml.safe_load(
            (ROOT / "experiments" / "experiment_matrix.yaml").read_text(
                encoding="utf-8"
            )
        )
        experiments = {item["id"]: item for item in matrix["experiments"]}

        subsets = experiments["D1_frozen_training_subsets"]
        self.assertEqual(subsets["status"], "completed")
        self.assertTrue((ROOT / subsets["evidence"]).is_file())

        base = experiments["B0_base_internal"]
        self.assertEqual(base["status"], "completed")
        self.assertTrue((ROOT / base["evidence"]).is_file())

        external = experiments["B1_base_external"]
        self.assertEqual(external["status"], "in_progress")
        self.assertIn("no auditable model result", external["progress"])

        sft_smoke = experiments["S0_sft_smoke"]
        self.assertEqual(sft_smoke["status"], "completed")
        self.assertTrue((ROOT / sft_smoke["evidence"]).is_file())
        self.assertIn("trainer_peak_allocated_delta_gib", sft_smoke["primary_metrics"])

        sft = experiments["S1_qlora_r16_domain"]
        self.assertEqual(sft["status"], "completed")
        for path in sft["evidence"].values():
            if path.startswith("experiments/results/"):
                self.assertTrue((ROOT / path).is_file(), path)

        smoke = experiments["R0_grpo_4090_smoke"]
        self.assertEqual(smoke["status"], "completed")
        self.assertTrue((ROOT / smoke["evidence"]).is_file())
        self.assertNotIn("peak_vram_gib", smoke["primary_metrics"])
        self.assertIn("adapter_updates", smoke["primary_metrics"])

        full_grpo = experiments["R1_grpo_48g_full"]
        self.assertEqual(full_grpo["status"], "completed")
        self.assertTrue((ROOT / full_grpo["evidence"]["summary"]).is_file())
        self.assertIn("result", full_grpo)
        self.assertIn("attribution", full_grpo)
        self.assertIn("protocol_deviation", full_grpo)

        reward_ablation = experiments["R2_reward_ablation"]
        self.assertEqual(reward_ablation["status"], "not_triggered")
        self.assertIn("did not establish", reward_ablation["result"])
        self.assertIn("was not run", reward_ablation["decision"])

    def test_public_docs_keep_external_and_measurement_boundaries_explicit(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        data_card = (ROOT / "docs" / "data_card.md").read_text(encoding="utf-8")
        runbook = (ROOT / "docs" / "autodl_runbook.md").read_text(encoding="utf-8")

        self.assertIn("ChartQA 与 ChartQAPro 数据准备门均已通过", readme)
        self.assertIn("Base 外部评测已在固定 2,500 + 1,946 条子集上启动", readme)
        self.assertIn("三组模型的外部评测和定性非退化检查仍待完成", readme)
        self.assertIn("Answer[-1]", data_card)
        self.assertIn("确定性排除且不回填", data_card)
        self.assertIn("selected source 与 evaluable 行数", runbook)


if __name__ == "__main__":
    unittest.main()
