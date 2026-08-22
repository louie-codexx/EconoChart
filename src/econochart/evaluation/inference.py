from __future__ import annotations

import time
from typing import Any

from PIL import Image

from econochart.config import ROOT
from econochart.data.training import system_prompt_for


def _device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _build_messages(record: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_for(record)}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": record["question"]},
            ],
        },
    ]


def generate_one(
    model: Any,
    processor: Any,
    record: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    image_path = (ROOT / record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image missing for {record['id']}: {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = _build_messages(record, image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_device(model))
    kwargs = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 320)),
        "do_sample": bool(generation_config.get("do_sample", False)),
        "repetition_penalty": float(generation_config.get("repetition_penalty", 1.0)),
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(generation_config.get("temperature", 0.8))
        kwargs["top_p"] = float(generation_config.get("top_p", 0.95))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    prompt_length = int(inputs["input_ids"].shape[1])
    completion_ids = generated[:, prompt_length:]
    prediction = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "prediction": prediction,
        "latency_seconds": round(latency, 6),
        "prompt_tokens": prompt_length,
        "completion_tokens": int(completion_ids.shape[1]),
    }
