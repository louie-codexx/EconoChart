from __future__ import annotations

import argparse
import json
from typing import Any

from econochart.config import load_config, resolve_model_path
from econochart.data.training import build_sft_dataset, load_record_sources, summarize_records
from econochart.models.loading import (
    build_lora_config,
    load_base_model,
    load_processor,
    trainable_parameter_summary,
)
from econochart.training.common import resolve_output_dir, save_run_snapshot, seed_everything, trainer_kwargs


def _attach_adapter(model: Any, adapter_config: dict[str, Any]) -> tuple[Any, Any | None]:
    peft_config = build_lora_config(adapter_config)
    if peft_config is None:
        _apply_full_tuning_policy(model, adapter_config)
        return model, None

    from peft import get_peft_model

    model = get_peft_model(model, peft_config)
    _apply_component_policy(model, adapter_config)
    return model, None


def _is_projector(name: str) -> bool:
    return any(value in name for value in ("projector", "merger", "multi_modal_projector"))


def _apply_component_policy(model: Any, adapter_config: dict[str, Any]) -> None:
    train_vision = bool(adapter_config.get("train_vision", False))
    train_projector = bool(adapter_config.get("train_projector", False))
    for name, parameter in model.named_parameters():
        normalized = name.lower()
        projector = _is_projector(normalized)
        vision = ("visual" in normalized or "vision" in normalized) and not projector
        if (vision and not train_vision) or (projector and not train_projector):
            parameter.requires_grad = False


def _apply_full_tuning_policy(model: Any, adapter_config: dict[str, Any]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    _apply_component_policy(model, adapter_config)


def run(
    config: dict[str, Any],
    *,
    model_override: str | None,
    output_override: str | None,
    resume: str | bool | None,
) -> None:
    from trl import SFTConfig, SFTTrainer

    seed = int(config.get("seed", 20260821))
    train_records = load_record_sources(config["data"]["train"], expected_split="train", seed=seed)
    eval_records = load_record_sources(config["data"]["eval"], expected_split="val", seed=seed)
    train_dataset = build_sft_dataset(train_records)
    eval_dataset = build_sft_dataset(eval_records)

    model_path = resolve_model_path(config, model_override)
    output_dir = resolve_output_dir(config, output_override)
    seed_everything(seed)
    model, quantized = load_base_model(model_path, config["model"], for_training=True)
    processor = load_processor(model_path, config["model"], padding_side="right")
    model, peft_config = _attach_adapter(model, config.get("adapter", {}))
    parameters = trainable_parameter_summary(model)
    if parameters["trainable_parameters"] == 0:
        raise RuntimeError("No trainable parameters remain after applying the tuning policy")

    training_values = trainer_kwargs(config, output_dir=output_dir)
    training_values.setdefault("seed", seed)
    training_values.setdefault("data_seed", seed)
    training_values.setdefault("max_length", None)
    training_values.setdefault("completion_only_loss", True)
    training_values.setdefault("packing", False)
    training_values.setdefault("padding_free", False)
    args = SFTConfig(**training_values)
    summaries = {"train": summarize_records(train_records), "eval": summarize_records(eval_records)}
    save_run_snapshot(
        output_dir,
        stage="sft",
        config=config,
        model_path=model_path,
        adapter_path=None,
        data_summary={**summaries, "quantized_base": quantized},
        parameter_summary=parameters,
    )
    print(json.dumps({"data": summaries, "parameters": parameters}, ensure_ascii=False, indent=2))

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    result = trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    if args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL SFT with LoRA, QLoRA, or full tuning.")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--model", help="Override base model path/ID (or set ECONOCHART_MODEL_PATH)")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--resume", nargs="?", const=True, help="Resume from a checkpoint path or latest checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resume = str(args.resume) if isinstance(args.resume, str) else (True if args.resume else None)
    run(
        load_config(args.config),
        model_override=args.model,
        output_override=args.output_dir,
        resume=resume,
    )


if __name__ == "__main__":
    main()
