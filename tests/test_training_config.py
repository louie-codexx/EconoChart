from __future__ import annotations

import unittest

from econochart.config import ConfigError, load_config
from econochart.training.common import validate_grpo_batch


class TrainingConfigTests(unittest.TestCase):
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
