from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def torch_dtype(name: str) -> Any:
    import torch

    normalized = name.lower().replace("torch.", "")
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[normalized]


def build_quantization_config(config: dict[str, Any]) -> Any | None:
    mode = str(config.get("mode", "none")).lower()
    if mode in {"none", "false", "off"}:
        return None
    if mode != "4bit":
        raise ValueError(f"Unsupported quantization mode: {mode}")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(config.get("quant_type", "nf4")),
        bnb_4bit_compute_dtype=torch_dtype(str(config.get("compute_dtype", "bfloat16"))),
        bnb_4bit_use_double_quant=bool(config.get("double_quant", True)),
    )


def load_processor(model_path: str, model_config: dict[str, Any], *, padding_side: str = "right") -> Any:
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {}
    if model_config.get("min_pixels") is not None:
        kwargs["min_pixels"] = int(model_config["min_pixels"])
    if model_config.get("max_pixels") is not None:
        kwargs["max_pixels"] = int(model_config["max_pixels"])
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def _model_class() -> Any:
    import transformers

    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
    ):
        candidate = getattr(transformers, name, None)
        if candidate is not None:
            return candidate
    raise ImportError(
        "This Transformers build does not expose a Qwen3-VL-compatible model class. "
        "Install the versions declared by `pip install -e '.[train]'`."
    )


def _single_process_device_map() -> dict[str, int] | None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    return {"": local_rank}


def load_base_model(
    model_path: str,
    model_config: dict[str, Any],
    *,
    for_training: bool,
) -> tuple[Any, bool]:
    quantization_config = build_quantization_config(model_config.get("quantization", {}))
    kwargs: dict[str, Any] = {
        "dtype": torch_dtype(str(model_config.get("dtype", "bfloat16"))),
        "low_cpu_mem_usage": True,
    }
    attention = model_config.get("attn_implementation")
    if attention:
        kwargs["attn_implementation"] = attention
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = _single_process_device_map()
    elif not for_training:
        kwargs["device_map"] = "auto"

    model = _model_class().from_pretrained(model_path, **kwargs)
    model.config.use_cache = not for_training
    if quantization_config is not None and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(model_config.get("gradient_checkpointing", True)),
        )
    return model, quantization_config is not None


def build_lora_config(adapter_config: dict[str, Any]) -> Any | None:
    method = str(adapter_config.get("method", "qlora")).lower()
    if method == "full":
        return None
    if method not in {"lora", "qlora"}:
        raise ValueError(f"adapter.method must be one of full/lora/qlora, got {method!r}")
    from peft import LoraConfig

    target_modules = adapter_config.get("target_modules")
    if target_modules == "all-linear":
        target_modules = "all-linear"
    elif not isinstance(target_modules, list) or not target_modules:
        raise ValueError("LoRA/QLoRA requires a non-empty target_modules list or 'all-linear'")
    return LoraConfig(
        r=int(adapter_config.get("r", 16)),
        lora_alpha=int(adapter_config.get("alpha", 32)),
        lora_dropout=float(adapter_config.get("dropout", 0.05)),
        target_modules=target_modules,
        bias=str(adapter_config.get("bias", "none")),
        task_type="CAUSAL_LM",
        use_rslora=bool(adapter_config.get("use_rslora", False)),
    )


def load_adapter(model: Any, adapter_path: str, *, trainable: bool = False) -> Any:
    from peft import PeftModel

    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {path}")
    return PeftModel.from_pretrained(model, str(path), is_trainable=trainable)


def load_trainable_adapter(model: Any, adapter_path: str) -> Any:
    return load_adapter(model, adapter_path, trainable=True)


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    component_parameters = {"vision": 0, "projector": 0, "language_or_other": 0}
    trainable_tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        normalized = name.lower()
        if any(value in normalized for value in ("projector", "merger", "multi_modal_projector")):
            component = "projector"
        elif "visual" in normalized or "vision" in normalized:
            component = "vision"
        else:
            component = "language_or_other"
        component_parameters[component] += parameter.numel()
    trainable = sum(component_parameters.values())
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100.0 * trainable / max(1, total), 6),
        "trainable_tensors": trainable_tensors,
        "trainable_parameters_by_component": component_parameters,
    }
