from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from econochart.config import ROOT, ConfigError, project_path


def resolve_output_dir(config: dict[str, Any], override: str | None = None) -> Path:
    value = override or config.get("training", {}).get("output_dir")
    if not value:
        raise ConfigError("training.output_dir is required")
    output_dir = project_path(value)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_grpo_batch(config: dict[str, Any], *, world_size: int | None = None) -> dict[str, int]:
    training = config.get("training", {})
    processes = world_size or int(os.getenv("WORLD_SIZE", "1"))
    batch = int(training.get("per_device_train_batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    generations = int(training.get("num_generations", 1))
    if min(processes, batch, accumulation, generations) < 1:
        raise ConfigError("GRPO batch, accumulation, world size, and num_generations must all be positive")
    effective_batch = processes * batch * accumulation
    if effective_batch % generations != 0:
        raise ConfigError(
            "GRPO requires world_size * per_device_train_batch_size * gradient_accumulation_steps "
            f"to be divisible by num_generations; got {processes} * {batch} * {accumulation} "
            f"= {effective_batch}, num_generations={generations}"
        )
    return {
        "world_size": processes,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "num_generations": generations,
        "prompts_per_update": effective_batch // generations,
    }


def _git_revision() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        result = {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def package_versions() -> dict[str, str | None]:
    names = ["econochart", "torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch
    except ImportError:
        return summary
    summary.update(
        {
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    )
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            summary["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return summary


def save_run_snapshot(
    output_dir: Path,
    *,
    stage: str,
    config: dict[str, Any],
    model_path: str,
    adapter_path: str | None,
    data_summary: dict[str, Any],
    parameter_summary: dict[str, Any] | None = None,
) -> Path:
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    snapshot = {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_revision(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "hardware": hardware_summary(),
        },
        "model": {"base": model_path, "adapter": adapter_path},
        "data": data_summary,
        "parameters": parameter_summary,
        "config": clean_config,
    }
    target = output_dir / "run_manifest.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(clean_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def trainer_kwargs(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    values = dict(config.get("training", {}))
    values["output_dir"] = str(output_dir)
    values.setdefault("report_to", ["tensorboard"])
    values.setdefault("run_name", output_dir.name)
    return values
