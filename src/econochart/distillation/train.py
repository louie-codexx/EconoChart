from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from econochart.config import ROOT, load_config, project_path, require, resolve_adapter_path, resolve_model_path
from econochart.data.training import system_prompt_for, user_prompt_for
from econochart.distillation.artifacts import (
    read_teacher_score_shard,
    record_identity,
    summarize_rollouts,
    validate_adapter_snapshot,
    validate_resume_snapshot,
    validate_rollout_row,
)
from econochart.distillation.losses import sparse_forward_kl_loss
from econochart.io import read_json, read_jsonl, sha256_file, write_json
from econochart.models.loading import (
    load_base_model,
    load_processor,
    load_trainable_adapter,
    trainable_parameter_summary,
)
from econochart.training.common import save_run_snapshot, seed_everything


def validate_step_budget(total_rows: int, accumulation_steps: int, expected_steps: int) -> int:
    if min(total_rows, accumulation_steps, expected_steps) < 1:
        raise ValueError("OPD rows, accumulation, and expected steps must be positive")
    if total_rows % accumulation_steps:
        raise ValueError(
            f"OPD micro-step count {total_rows} must be divisible by gradient accumulation {accumulation_steps}"
        )
    actual = total_rows // accumulation_steps
    if actual != expected_steps:
        raise ValueError(f"OPD expected {expected_steps} optimizer steps, derived {actual}")
    return actual


def align_training_artifacts(
    records: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    round_id: int,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    record_map = {str(record["id"]): record for record in records}
    score_map = {str(score["rollout_key"]): score for score in scores}
    if len(record_map) != len(records):
        raise ValueError("OPD training prompt source contains duplicate IDs")
    if len(score_map) != len(scores):
        raise ValueError("OPD teacher scores contain duplicate rollout keys")
    aligned = []
    for rollout in rollouts:
        validate_rollout_row(rollout)
        if int(rollout["round"]) != round_id:
            raise ValueError(f"OPD training received round {rollout['round']} rollout, expected {round_id}")
        record = record_map.get(str(rollout["record_id"]))
        if record is None or rollout["record_sha256"] != record_identity(record):
            raise ValueError(f"OPD training source identity mismatch for {rollout['record_id']}")
        score = score_map.get(str(rollout["rollout_key"]))
        if score is None:
            raise ValueError(f"Missing teacher score for {rollout['rollout_key']}")
        for field in ("prompt_token_ids_sha256", "completion_token_ids_sha256", "completion_token_ids"):
            if score.get(field) != rollout.get(field):
                raise ValueError(f"Teacher score field {field} differs for {rollout['rollout_key']}")
        aligned.append((record, rollout, score))
    unknown_scores = sorted(score_map.keys() - {str(row["rollout_key"]) for row in rollouts})
    if unknown_scores:
        raise ValueError(f"Teacher scores contain unknown rollout keys; first={unknown_scores[:3]}")
    return aligned


def _load_records(source: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    path = project_path(require(source, "path"))
    if not path.is_file():
        raise FileNotFoundError(f"OPD training source does not exist: {path}")
    records = list(read_jsonl(path))
    expected_rows = int(require(source, "expected_rows"))
    if len(records) != expected_rows:
        raise ValueError(f"OPD training source expected {expected_rows} rows, found {len(records)}")
    expected_split = str(require(source, "expected_split"))
    wrong = [str(row.get("id", "<missing>")) for row in records if row.get("split") != expected_split]
    if expected_split != "train" or wrong:
        raise ValueError(f"OPD training source must contain only split=train; first={wrong[:1]}")
    return path, records


def _load_rollouts(source: dict[str, Any]) -> tuple[Path, Path, list[dict[str, Any]]]:
    path = project_path(require(source, "path"))
    manifest_path = project_path(require(source, "manifest"))
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"OPD rollout artifacts are missing: {path}, {manifest_path}")
    rows = list(read_jsonl(path))
    expected = int(require(source, "expected_rows"))
    if len(rows) != expected:
        raise ValueError(f"OPD training expected {expected} rollouts, found {len(rows)}")
    summarize_rollouts(rows)
    manifest = read_json(manifest_path)
    if manifest.get("rollouts", {}).get("sha256") != sha256_file(path):
        raise ValueError("OPD rollout manifest hash does not match the rollout file")
    return path, manifest_path, rows


def _load_scores(source: dict[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = project_path(require(source, "manifest"))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"OPD teacher score manifest does not exist: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("gate") != "OPD_TEACHER_SCORE_ARTIFACT_PASS" or manifest.get("status") != "passed":
        raise ValueError("OPD teacher score manifest did not pass its artifact gate")
    rows: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        recorded_path = Path(str(shard["path"])).expanduser().resolve()
        portable_path = (manifest_path.parent / recorded_path.name).resolve()
        path = recorded_path if recorded_path.is_file() else portable_path
        if not path.is_file():
            raise FileNotFoundError(
                "OPD teacher score shard is missing at both its recorded and portable locations: "
                f"{recorded_path}, {portable_path}"
            )
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"OPD teacher score shard hash changed: {path}")
        rows.extend(read_teacher_score_shard(path))
    expected = int(require(source, "expected_rows"))
    if len(rows) != expected:
        raise ValueError(f"OPD training expected {expected} teacher score rows, found {len(rows)}")
    return manifest_path, rows, manifest


def _require_gate(entry: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = project_path(require(entry, "path"))
    if not path.is_file():
        raise FileNotFoundError(f"Missing OPD {label} prerequisite report: {path}")
    report = read_json(path)
    expected_gate = str(require(entry, "expected_gate"))
    if report.get("gate") != expected_gate or report.get("status") != "passed":
        raise ValueError(
            f"OPD {label} prerequisite failed: expected {expected_gate}/passed, "
            f"found {report.get('gate')}/{report.get('status')}"
        )
    return {"path": str(path), "sha256": sha256_file(path), "gate": expected_gate}


def _messages(record: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt_for(record)}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt_for(record)},
            ],
        },
    ]


def _device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _build_model_inputs(
    processor: Any,
    model: Any,
    record: dict[str, Any],
    rollout: dict[str, Any],
    image_hash_cache: dict[str, str],
) -> tuple[dict[str, Any], int]:
    import torch

    image_path = (ROOT / str(record["image"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"OPD training image missing for {record['id']}: {image_path}")
    image_key = str(image_path)
    actual_image_sha256 = image_hash_cache.get(image_key)
    if actual_image_sha256 is None:
        actual_image_sha256 = sha256_file(image_path)
        image_hash_cache[image_key] = actual_image_sha256
    if actual_image_sha256 != rollout.get("image_sha256"):
        raise ValueError(f"OPD training image content changed for {record['id']}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor.apply_chat_template(
        _messages(record, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
    if prompt_ids != rollout["prompt_token_ids"]:
        raise ValueError(f"Student processor prompt tokens changed for {rollout['rollout_key']}")
    completion_ids = [int(value) for value in rollout["completion_token_ids"]]
    completion_tensor = torch.tensor([completion_ids], dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], completion_tensor], dim=1)
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(completion_tensor)], dim=1
    )
    device = _device(model)
    return (
        {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(inputs).items()},
        len(completion_ids),
    )


def _build_optimizer(model: Any, training: dict[str, Any]) -> Any:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("OPD student has no trainable parameters")
    optimizer_name = str(training.get("optim", "paged_adamw_8bit"))
    kwargs = {
        "lr": float(require(training, "learning_rate")),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if optimizer_name == "paged_adamw_8bit":
        from bitsandbytes.optim import PagedAdamW8bit

        return PagedAdamW8bit(parameters, **kwargs)
    if optimizer_name == "adamw_torch":
        import torch

        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unsupported OPD optimizer: {optimizer_name}")


def _build_scheduler(optimizer: Any, training: dict[str, Any], total_steps: int) -> Any:
    from transformers import get_scheduler

    warmup_steps = round(total_steps * float(training.get("warmup_ratio", 0.0)))
    return get_scheduler(
        str(training.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _disable_dropout(model: Any) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def _save_checkpoint(
    output_dir: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    optimizer_step: int,
    next_micro_step: int,
) -> Path:
    import torch

    checkpoint = output_dir / f"checkpoint-{optimizer_step:04d}"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite OPD checkpoint: {checkpoint}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint.name}.incomplete-", dir=output_dir))
    model.save_pretrained(temporary / "adapter")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    write_json(
        temporary / "state.json",
        {"optimizer_step": optimizer_step, "next_micro_step": next_micro_step},
    )
    temporary.replace(checkpoint)
    return checkpoint


def _resume_state(checkpoint_value: str | None) -> tuple[Path | None, dict[str, int]]:
    if not checkpoint_value:
        return None, {"optimizer_step": 0, "next_micro_step": 0}
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    state_path = checkpoint / "state.json"
    if not (checkpoint / "adapter").is_dir() or not state_path.is_file():
        raise FileNotFoundError(f"Invalid OPD resume checkpoint: {checkpoint}")
    state = read_json(state_path)
    return checkpoint, {
        "optimizer_step": int(state["optimizer_step"]),
        "next_micro_step": int(state["next_micro_step"]),
    }


def validate_resume_position(
    state: dict[str, int],
    *,
    accumulation_steps: int,
    total_micro_steps: int,
    total_optimizer_steps: int,
) -> None:
    optimizer_step = state["optimizer_step"]
    next_micro_step = state["next_micro_step"]
    if optimizer_step < 0 or optimizer_step > total_optimizer_steps:
        raise ValueError("OPD resume optimizer step is outside the configured range")
    if next_micro_step < 0 or next_micro_step > total_micro_steps:
        raise ValueError("OPD resume micro-step is outside the configured range")
    if next_micro_step != optimizer_step * accumulation_steps:
        raise ValueError(
            "OPD resume checkpoint is not on a gradient-accumulation boundary: "
            f"optimizer_step={optimizer_step}, next_micro_step={next_micro_step}"
        )


def run(
    config: dict[str, Any],
    *,
    model_override: str | None,
    adapter_override: str | None,
    output_override: str | None,
    resume_checkpoint: str | None,
) -> dict[str, Any]:
    import torch

    seed = int(config.get("seed", 20260821))
    round_id = int(require(config, "experiment.round"))
    prompt_path, records = _load_records(require(config, "data.prompts"))
    rollout_path, rollout_manifest_path, rollouts = _load_rollouts(require(config, "data.rollouts"))
    score_manifest_path, scores, score_manifest = _load_scores(require(config, "data.teacher_scores"))
    aligned = align_training_artifacts(records, rollouts, scores, round_id=round_id)
    prerequisites = {
        label: _require_gate(entry, label=label)
        for label, entry in require(config, "prerequisites").items()
    }
    distillation = require(config, "distillation")
    if int(require(distillation, "top_k")) != int(score_manifest["scoring"]["top_k"]):
        raise ValueError("Student distillation top_k differs from the teacher score artifact")
    if float(require(distillation, "temperature")) != float(score_manifest["scoring"]["temperature"]):
        raise ValueError("Student distillation temperature differs from the teacher score artifact")
    if float(score_manifest["scoring"]["mean_topk_mass"]) < float(
        require(distillation, "minimum_mean_topk_mass")
    ):
        raise ValueError("Teacher score top-k mass is below the student training gate")

    training = require(config, "training")
    checkpoint, state = _resume_state(resume_checkpoint)
    expected_micro_steps = int(require(training, "expected_micro_steps"))
    if len(aligned) != expected_micro_steps:
        raise ValueError(f"OPD expected {expected_micro_steps} aligned rows, found {len(aligned)}")
    accumulation = int(require(training, "gradient_accumulation_steps"))
    total_optimizer_steps = validate_step_budget(
        len(aligned), accumulation, int(require(training, "expected_optimizer_steps"))
    )
    validate_resume_position(
        state,
        accumulation_steps=accumulation,
        total_micro_steps=len(aligned),
        total_optimizer_steps=total_optimizer_steps,
    )
    if int(training.get("num_train_epochs", 1)) != 1:
        raise ValueError("OPD v1 requires exactly one epoch so each frozen rollout is consumed once")
    if int(training.get("per_device_train_batch_size", 1)) != 1:
        raise ValueError("OPD v1 currently supports per_device_train_batch_size=1 only")

    output_dir = project_path(output_override or require(training, "output_dir"))
    if output_dir.exists() and any(output_dir.iterdir()) and checkpoint is None:
        raise FileExistsError(f"OPD student output is non-empty; use an audited checkpoint to resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter = output_dir / str(require(training, "final_adapter_dir"))
    if final_adapter.exists():
        raise FileExistsError(f"OPD final adapter already exists: {final_adapter}")

    model_path = resolve_model_path(config, model_override)
    source_adapter = resolve_adapter_path(config, adapter_override)
    if not source_adapter:
        raise ValueError("OPD student update requires the frozen GRPO adapter")
    source_adapter_identity = validate_adapter_snapshot(
        source_adapter,
        str(require(config, "model.expected_adapter_sha256")),
    )
    resume_data = {
        "prompts": {"path": str(prompt_path), "sha256": sha256_file(prompt_path), "rows": len(records)},
        "rollouts": {"path": str(rollout_path), "sha256": sha256_file(rollout_path), "rows": len(rollouts)},
        "rollout_manifest_sha256": sha256_file(rollout_manifest_path),
        "teacher_score_manifest": {
            "path": str(score_manifest_path),
            "sha256": sha256_file(score_manifest_path),
            "rows": len(scores),
        },
        "student_source_adapter": source_adapter_identity,
        "prerequisites": prerequisites,
    }
    if checkpoint is not None:
        if checkpoint.parent.resolve() != output_dir.resolve():
            raise ValueError(f"OPD resume checkpoint must be inside its output directory: {output_dir}")
        validate_resume_snapshot(
            output_dir,
            stage="opd_student_update",
            config=config,
            model_path=model_path,
            adapter_path=source_adapter,
            data_summary=resume_data,
        )
    adapter_to_load = str(checkpoint / "adapter") if checkpoint is not None else source_adapter
    seed_everything(seed)
    model, quantized = load_base_model(model_path, config["model"], for_training=True)
    if not quantized:
        raise ValueError("Formal 48GB OPD student update requires the registered 4-bit base")
    model = load_trainable_adapter(model, adapter_to_load)
    if bool(training.get("disable_dropout", True)):
        _disable_dropout(model)
    model.train()
    processor = load_processor(model_path, config["model"], padding_side="left")
    parameters = trainable_parameter_summary(model)
    optimizer = _build_optimizer(model, training)
    scheduler = _build_scheduler(optimizer, training, total_optimizer_steps)
    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
        scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True))
    optimizer.zero_grad(set_to_none=True)
    if bool(training.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if checkpoint is None:
        save_run_snapshot(
            output_dir,
            stage="opd_student_update",
            config=config,
            model_path=model_path,
            adapter_path=source_adapter,
            data_summary=resume_data,
            parameter_summary=parameters,
        )

    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    temperature = float(require(distillation, "temperature"))
    logging_steps = int(training.get("logging_steps", 10))
    save_steps = int(training.get("save_steps", 100))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    optimizer_step = state["optimizer_step"]
    next_micro_step = state["next_micro_step"]
    image_hash_cache: dict[str, str] = {}
    losses: list[float] = []
    for micro_index in range(next_micro_step, len(aligned)):
        record, rollout, score = aligned[micro_index]
        inputs, completion_length = _build_model_inputs(
            processor,
            model,
            record,
            rollout,
            image_hash_cache,
        )
        forward_kwargs: dict[str, Any] = {**inputs, "use_cache": False}
        if supports_logits_to_keep:
            forward_kwargs["logits_to_keep"] = completion_length + 1
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=bool(training.get("bf16", True)) and torch.cuda.is_available(),
        ):
            outputs = model(**forward_kwargs)
            if supports_logits_to_keep:
                student_logits = outputs.logits[:, :-1, :]
            else:
                start = inputs["input_ids"].shape[1] - completion_length - 1
                student_logits = outputs.logits[:, start : start + completion_length, :]
            teacher_ids = torch.tensor(score["topk_token_ids"], device=student_logits.device).unsqueeze(0)
            teacher_logprobs = torch.tensor(
                score["topk_logprobs"], device=student_logits.device, dtype=torch.float32
            ).unsqueeze(0)
            loss = sparse_forward_kl_loss(
                student_logits,
                teacher_ids,
                teacher_logprobs,
                temperature=temperature,
            )
        raw_loss = float(loss.detach().item())
        losses.append(raw_loss)
        (loss / accumulation).backward()
        if (micro_index + 1) % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        if optimizer_step % logging_steps == 0:
            recent = losses[-logging_steps * accumulation :]
            print(
                json.dumps(
                    {
                        "optimizer_step": optimizer_step,
                        "micro_step": micro_index + 1,
                        "loss": round(sum(recent) / len(recent), 6),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
            )
        if optimizer_step % save_steps == 0 and optimizer_step < total_optimizer_steps:
            _save_checkpoint(
                output_dir,
                model,
                optimizer,
                scheduler,
                optimizer_step=optimizer_step,
                next_micro_step=micro_index + 1,
            )

    if optimizer_step != total_optimizer_steps:
        raise ValueError(f"OPD optimizer step mismatch: expected {total_optimizer_steps}, found {optimizer_step}")
    model.save_pretrained(final_adapter)
    processor.save_pretrained(final_adapter)
    adapter_weights = final_adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"OPD final adapter weights were not saved: {adapter_weights}")
    result = {
        "status": "completed",
        "gate": "OPD_ROUND1_STUDENT_UPDATE_COMPLETE",
        "experiment": config.get("experiment", {}),
        "micro_steps": len(aligned),
        "optimizer_steps": optimizer_step,
        "resumed_from_micro_step": next_micro_step,
        "processed_micro_steps_this_process": len(losses),
        "mean_loss_this_process": round(sum(losses) / len(losses), 8) if losses else None,
        "final_learning_rate": scheduler.get_last_lr()[0],
        "supports_logits_to_keep": supports_logits_to_keep,
        "source_adapter": source_adapter_identity,
        "final_adapter": str(final_adapter),
        "final_adapter_sha256": sha256_file(adapter_weights),
        "parameters": parameters,
        "post_training": config.get("post_training", {}),
    }
    write_json(output_dir / str(require(training, "result_file")), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update the Qwen3-VL student from frozen multimodal OPD scores.")
    parser.add_argument("--config", default="configs/opd/round1_student_train_v1.yaml")
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Audited checkpoint directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        load_config(args.config),
        model_override=args.model,
        adapter_override=args.adapter,
        output_override=args.output_dir,
        resume_checkpoint=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
