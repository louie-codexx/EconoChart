from __future__ import annotations

import unittest
from pathlib import Path

from econochart.config import ConfigError, load_config
from econochart.evaluation.runner import _load_eval_records
from econochart.models.loading import trainable_parameter_summary
from econochart.preflight import REQUIRED_PACKAGES, run_preflight
from econochart.training.common import validate_grpo_batch


class _Parameter:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class Params4bit(_Parameter):
    def element_size(self) -> int:
        return 1


class _Model:
    def __init__(self) -> None:
        self._parameters = [
            ("base_model.language.weight", Params4bit(100, requires_grad=False)),
            ("base_model.embed.weight", _Parameter(20, requires_grad=False)),
            ("language.lora_A.default.weight", _Parameter(10, requires_grad=True)),
            ("visual.lora_A.default.weight", _Parameter(5, requires_grad=True)),
        ]

    def named_parameters(self):
        return iter(self._parameters)

    def parameters(self):
        return (parameter for _, parameter in self._parameters)


class TrainingConfigTests(unittest.TestCase):
    def test_model_runtime_preflight_requires_safe_torch_minimum(self) -> None:
        for stage in ("sft", "grpo", "eval"):
            self.assertEqual(REQUIRED_PACKAGES[stage]["torch"], ">=2.6")
            self.assertEqual(REQUIRED_PACKAGES[stage]["torchvision"], ">=0.21")

        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"torch>=2.6"', pyproject)
        self.assertIn('"torchvision>=0.21"', pyproject)

    def test_parameter_summary_restores_packed_4bit_logical_count(self) -> None:
        summary = trainable_parameter_summary(_Model())
        self.assertEqual(summary["trainable_parameters"], 15)
        self.assertEqual(summary["total_parameters"], 235)
        self.assertAlmostEqual(summary["trainable_percent"], 100 * 15 / 235, places=6)
        self.assertEqual(
            summary["trainable_parameters_by_component"],
            {"vision": 5, "projector": 0, "language_or_other": 10},
        )

    def test_shipped_grpo_configs_have_valid_effective_batches(self) -> None:
        for path in (
            "configs/train/grpo_qlora_4090_smoke.yaml",
            "configs/train/grpo_qlora_48g.yaml",
        ):
            summary = validate_grpo_batch(load_config(path), world_size=1)
            self.assertEqual(summary["effective_batch_size"] % summary["num_generations"], 0)
            self.assertGreaterEqual(summary["prompts_per_update"], 1)

    def test_formal_grpo_config_locks_the_48g_run_budget(self) -> None:
        config = load_config("configs/train/grpo_qlora_48g.yaml")
        training = config["training"]
        self.assertEqual(config["data"]["train"][0]["expected_rows"], 3600)
        self.assertEqual(config["data"]["eval"][0]["expected_rows"], 512)
        self.assertEqual(training["num_generations"], 4)
        self.assertEqual(training["gradient_accumulation_steps"], 4)
        self.assertEqual(training["learning_rate"], 0.000005)
        self.assertEqual(training["beta"], 0.001)
        self.assertEqual(training["save_steps"], 100)

    def test_invalid_grpo_batch_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            validate_grpo_batch(
                {
                    "training": {
                        "per_device_train_batch_size": 1,
                        "gradient_accumulation_steps": 2,
                        "num_generations": 3,
                    }
                },
                world_size=1,
            )

    def test_ablation_config_inherits_complete_base(self) -> None:
        config = load_config("configs/train/sft_qlora_r8_ablation.yaml")
        self.assertEqual(config["adapter"]["r"], 8)
        self.assertEqual(config["adapter"]["target_modules"][0], "q_proj")
        self.assertEqual(config["model"]["quantization"]["mode"], "4bit")
        self.assertEqual(config["training"]["learning_rate"], 0.0001)

    def test_formal_sft_config_locks_selected_hyperparameters(self) -> None:
        config = load_config("configs/train/sft_qlora_4090.yaml")
        self.assertEqual(config["adapter"]["r"], 16)
        self.assertEqual(config["adapter"]["alpha"], 32)
        self.assertEqual(config["training"]["learning_rate"], 0.0002)
        self.assertEqual(config["training"]["num_train_epochs"], 2)
        self.assertEqual(config["data"]["train"][0]["expected_rows"], 9600)

        central_screen = load_config("configs/train/sft_qlora_r16_screen.yaml")
        self.assertEqual(central_screen["training"]["learning_rate"], 0.0001)

    def test_public_mix_sft_changes_only_the_registered_data_factor(self) -> None:
        control = load_config("configs/train/sft_qlora_4090.yaml")
        mixed = load_config("configs/train/sft_qlora_4090_mixed_chartqa.yaml")

        self.assertEqual(mixed["model"], control["model"])
        self.assertEqual(mixed["adapter"], control["adapter"])
        self.assertEqual(mixed["seed"], control["seed"])
        for key, value in control["training"].items():
            if key != "output_dir":
                self.assertEqual(mixed["training"][key], value, key)
        self.assertEqual(mixed["data"]["train"][0], control["data"]["train"][0])
        self.assertEqual(mixed["data"]["train"][1]["expected_rows"], 3200)
        self.assertEqual(mixed["data"]["train"][1]["max_samples"], 3200)
        self.assertEqual(mixed["data"]["train"][1]["sampling_namespace"], "chartqa_train_s5_v1")
        self.assertEqual(mixed["data"]["expected_totals"], {"train": 12800, "eval": 512})
        self.assertEqual(
            mixed["experiment"]["registered_decision"]["restoration_target"],
            "The ChartQA exact-match point estimate must recover at least 0.0126 of the 0.0252 Base-to-SFT loss.",
        )
        self.assertNotEqual(mixed["training"]["output_dir"], control["training"]["output_dir"])

    def test_inputs_only_preflight_audits_data_without_claiming_launch_readiness(self) -> None:
        report = run_preflight(
            load_config("configs/train/sft_qlora_smoke.yaml"),
            stage="sft",
            inputs_only=True,
        )

        self.assertEqual(report["status"], "passed", report["issues"])
        self.assertEqual(report["scope"], "inputs_only")
        self.assertNotIn("packages", report["checks"])
        self.assertNotIn("cuda", report["checks"])
        self.assertNotIn("model", report["checks"])
        self.assertFalse(report["checks"]["launch_readiness"]["assessed"])
        self.assertEqual(report["checks"]["data"]["train"]["selected"]["rows"], 8)
        self.assertEqual(report["checks"]["data"]["eval"]["selected"]["rows"], 4)
        self.assertEqual(report["checks"]["data"]["train"]["selected"]["missing_image_count"], 0)
        self.assertEqual(
            report["checks"]["data_leakage"]["train_to_eval"]["id_overlap_count"],
            0,
        )

    def test_rank_candidates_match_each_selected_learning_rate(self) -> None:
        pairs = (
            (
                "configs/train/sft_qlora_r8_lr5e5_screen.yaml",
                "configs/train/sft_qlora_lr5e5_ablation.yaml",
                0.00005,
            ),
            (
                "configs/train/sft_qlora_r8_ablation.yaml",
                "configs/train/sft_qlora_r16_screen.yaml",
                0.0001,
            ),
            (
                "configs/train/sft_qlora_r8_lr2e4_screen.yaml",
                "configs/train/sft_qlora_lr2e4_ablation.yaml",
                0.0002,
            ),
        )
        for rank8_path, rank16_path, learning_rate in pairs:
            rank8 = load_config(rank8_path)
            rank16 = load_config(rank16_path)
            self.assertEqual(rank8["adapter"]["r"], 8)
            self.assertEqual(rank16["adapter"]["r"], 16)
            self.assertEqual(rank8["adapter"]["alpha"], 16)
            self.assertEqual(rank16["adapter"]["alpha"], 32)
            self.assertEqual(rank8["training"]["learning_rate"], learning_rate)
            self.assertEqual(rank16["training"]["learning_rate"], learning_rate)
            self.assertEqual(rank8["training"]["num_train_epochs"], 1)
            self.assertEqual(rank16["training"]["num_train_epochs"], 1)
            self.assertEqual(
                rank8["data"]["train"][0]["path"],
                rank16["data"]["train"][0]["path"],
            )
            self.assertEqual(rank8["data"]["train"][0]["expected_rows"], 4800)
            self.assertEqual(rank16["data"]["train"][0]["expected_rows"], 4800)

    def test_development_evaluation_explicitly_accepts_validation_split(self) -> None:
        config = {
            "seed": 20260821,
            "data": {
                "test": [
                    {
                        "path": "data/samples/econochart_v2/annotations/val.jsonl",
                        "expected_rows": 8,
                    }
                ]
            },
            "evaluation": {"expected_split": "val"},
        }
        records = _load_eval_records(config)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(row["split"] == "val" for row in records))

    def test_formal_and_development_configs_freeze_their_row_counts(self) -> None:
        base = load_config("configs/eval/base_internal.yaml")
        formal = load_config("configs/eval/sft_internal.yaml")
        grpo = load_config("configs/eval/grpo_internal.yaml")
        development = load_config("configs/eval/development_val_512.yaml")
        self.assertEqual(base["data"]["test"][0]["expected_rows"], 2496)
        self.assertEqual(formal["experiment"]["checkpoint_stage"], "sft")
        self.assertEqual(formal["model"]["adapter_path"], "outputs/sft_qlora_r16_domain_v1/final_adapter")
        self.assertEqual(formal["data"]["test"], base["data"]["test"])
        self.assertEqual(formal["generation"], base["generation"])
        self.assertEqual(formal["evaluation"]["expected_split"], "test")
        self.assertEqual(grpo["experiment"]["checkpoint_stage"], "grpo")
        self.assertEqual(grpo["model"]["adapter_path"], "outputs/grpo_qlora_48g_domain_v1/final_adapter")
        self.assertEqual(grpo["data"]["test"], formal["data"]["test"])
        self.assertEqual(grpo["generation"], formal["generation"])
        self.assertEqual(grpo["evaluation"], formal["evaluation"])
        self.assertEqual(development["evaluation"]["expected_split"], "val")
        self.assertEqual(development["data"]["test"][0]["expected_rows"], 512)

    def test_external_and_mixed_evaluations_share_frozen_records_and_generation(self) -> None:
        base = load_config("configs/eval/base_external.yaml")
        sft = load_config("configs/eval/sft_external.yaml")
        grpo = load_config("configs/eval/grpo_external.yaml")
        mixed = load_config("configs/eval/sft_mixed_external.yaml")

        self.assertEqual([source["expected_rows"] for source in base["data"]["test"]], [2500, 1946])
        self.assertEqual(base["data"]["expected_totals"]["test"], 4446)
        for candidate in (sft, grpo, mixed):
            self.assertEqual(candidate["data"], base["data"])
            self.assertEqual(candidate["generation"], base["generation"])
            self.assertEqual(candidate["evaluation"], base["evaluation"])
            self.assertNotEqual(candidate["model"]["adapter_path"], base["model"]["adapter_path"])
            self.assertNotEqual(candidate["training"]["output_dir"], base["training"]["output_dir"])

        internal = load_config("configs/eval/sft_internal.yaml")
        mixed_internal = load_config("configs/eval/sft_mixed_internal.yaml")
        self.assertEqual(mixed_internal["data"], internal["data"])
        self.assertEqual(mixed_internal["generation"], internal["generation"])
        self.assertEqual(mixed_internal["evaluation"], internal["evaluation"])
        self.assertEqual(
            mixed_internal["model"]["adapter_path"],
            "outputs/sft_qlora_r16_domain_chartqa_v1/final_adapter",
        )

    def test_mmefinance_evaluations_are_open_answer_eval_only_and_paired(self) -> None:
        base = load_config("configs/eval/base_mmefinance.yaml")
        sft = load_config("configs/eval/sft_mmefinance.yaml")

        self.assertEqual(base["experiment"]["benchmark"], "mmefinance_en_open")
        self.assertTrue(base["experiment"]["evaluation_only"])
        self.assertEqual(base["data"]["expected_totals"], {"test": 1171})
        self.assertEqual(base["data"]["test"][0]["expected_rows"], 1171)
        self.assertEqual(base["generation"]["max_new_tokens"], 512)
        self.assertFalse(base["generation"]["do_sample"])
        self.assertEqual(sft["data"], base["data"])
        self.assertEqual(sft["generation"], base["generation"])
        self.assertEqual(sft["evaluation"], base["evaluation"])
        self.assertIsNone(base["model"]["adapter_path"])
        self.assertEqual(sft["experiment"]["checkpoint_stage"], "sft_mixed")
        self.assertEqual(
            sft["model"]["adapter_path"],
            "outputs/sft_qlora_r16_domain_chartqa_v1/final_adapter",
        )
        self.assertNotEqual(sft["training"]["output_dir"], base["training"]["output_dir"])


if __name__ == "__main__":
    unittest.main()
