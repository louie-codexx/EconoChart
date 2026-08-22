from __future__ import annotations

import argparse
import json
from typing import Any

from econochart.config import ConfigError, load_config, resolve_adapter_path, resolve_model_path
from econochart.data.training import build_grpo_dataset, load_record_sources, summarize_records
from econochart.models.loading import (
    load_base_model,
    load_processor,
    load_trainable_adapter,
    trainable_parameter_summary,
)
from econochart.rewards.components import REWARD_FUNCTIONS
from econochart.training.common import (
    resolve_output_dir,
    save_run_snapshot,
    seed_everything,
    trainer_kwargs,
    validate_grpo_batch,
)


def run(
    config: dict[str, Any],
    *,
    model_override: str | None,
    adapter_override: str | None,
    output_override: str | None,
    resume: str | bool | None,
) -> None:
    from trl import GRPOConfig, GRPOTrainer

    batch_summary = validate_grpo_batch(config)
    seed = int(config.get("seed", 20260821))
    train_records = load_record_sources(config["data"]["train"], expected_split="train", seed=seed)
    eval_records = load_record_sources(config["data"]["eval"], expected_split="val", seed=seed)
    train_dataset = build_grpo_dataset(train_records)
    eval_dataset = build_grpo_dataset(eval_records)

    model_path = resolve_model_path(config, model_override)
    adapter_path = resolve_adapter_path(config, adapter_override)
    if not adapter_path:
        raise ConfigError(
            "GRPO must start from an evaluated SFT adapter. Set ECONOCHART_ADAPTER_PATH, "
            "model.adapter_path, or --adapter."
        )
    output_dir = resolve_output_dir(config, output_override)
    seed_everything(seed)
    model, quantized = load_base_model(model_path, config["model"], for_training=True)
    model = load_trainable_adapter(model, adapter_path)
    processor = load_processor(model_path, config["model"], padding_side="left")
    parameters = trainable_parameter_summary(model)
    if parameters["trainable_parameters"] == 0:
        raise RuntimeError("The SFT adapter was loaded without trainable parameters")

    training_values = trainer_kwargs(config, output_dir=output_dir)
    training_values.setdefault("seed", seed)
    training_values.setdefault("data_seed", seed)
    training_values.setdefault("remove_unused_columns", False)
    reward_weights = config.get("rewards", {}).get("weights", [0.35, 0.20, 0.15, 0.15, 0.10, 0.05])
    if len(reward_weights) != len(REWARD_FUNCTIONS):
        raise ConfigError(
            f"rewards.weights must have {len(REWARD_FUNCTIONS)} values in numeric/trend/format/evidence/risk/length order"
        )
    training_values["reward_weights"] = [float(value) for value in reward_weights]
    args = GRPOConfig(**training_values)
    summaries = {
        "train": summarize_records(train_records),
        "eval": summarize_records(eval_records),
        "grpo_train_rows": len(train_dataset),
        "grpo_eval_rows": len(eval_dataset),
        "batch": batch_summary,
        "quantized_base": quantized,
    }
    save_run_snapshot(
        output_dir,
        stage="grpo",
        config=config,
        model_path=model_path,
        adapter_path=adapter_path,
        data_summary=summaries,
        parameter_summary=parameters,
    )
    print(json.dumps({"data": summaries, "parameters": parameters}, ensure_ascii=False, indent=2))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCTIONS,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run verifiable multimodal GRPO from an SFT adapter.")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--model", help="Override base model path/ID (or set ECONOCHART_MODEL_PATH)")
    parser.add_argument("--adapter", help="SFT adapter path (or set ECONOCHART_ADAPTER_PATH)")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--resume", nargs="?", const=True, help="Resume from a checkpoint path or latest checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        load_config(args.config),
        model_override=args.model,
        adapter_override=args.adapter,
        output_override=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
