from __future__ import annotations

import unittest

from econochart.config import ConfigError, load_config
from econochart.models.loading import trainable_parameter_summary
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


if __name__ == "__main__":
    unittest.main()
