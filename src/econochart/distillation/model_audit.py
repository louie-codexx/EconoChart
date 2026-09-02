from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from econochart.io import sha256_file, write_json

TOKENIZER_AND_PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


def _combined_fingerprint(entries: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(entries.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str, issues: list[str]) -> dict[str, Any]:
    if not path.is_file():
        issues.append(f"missing {label}: {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(f"invalid {label} {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"invalid {label} root in {path.name}: expected object")
        return {}
    return payload


def _validate_safetensors_header(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return len(list(handle.keys()))


def audit_model_snapshot(
    model_dir: str | Path,
    *,
    expected_shards: int | None = None,
    minimum_weight_gib: float = 0.0,
    expected_model_type: str | None = None,
    validate_safetensors: bool = False,
    hash_shards: bool = False,
) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model snapshot directory does not exist: {root}")

    issues: list[str] = []
    config_path = root / "config.json"
    config = _read_mapping(config_path, label="model config", issues=issues)
    model_type = str(config.get("model_type", ""))
    architectures = [str(value) for value in config.get("architectures", [])]
    if expected_model_type and model_type != expected_model_type:
        issues.append(f"model_type mismatch: expected {expected_model_type}, found {model_type or '<missing>'}")

    index_path = root / "model.safetensors.index.json"
    index = _read_mapping(index_path, label="safetensors index", issues=issues)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        issues.append("safetensors index has no non-empty weight_map")
        weight_map = {}
    referenced_shards = sorted({str(value) for value in weight_map.values()})
    if expected_shards is not None and len(referenced_shards) != expected_shards:
        issues.append(f"shard count mismatch: expected {expected_shards}, found {len(referenced_shards)}")

    shards: list[dict[str, Any]] = []
    total_weight_bytes = 0
    for name in referenced_shards:
        path = root / name
        entry: dict[str, Any] = {"name": name, "exists": path.is_file()}
        if not path.is_file():
            issues.append(f"missing referenced shard: {name}")
            shards.append(entry)
            continue
        size = path.stat().st_size
        total_weight_bytes += size
        entry["bytes"] = size
        if size <= 0:
            issues.append(f"empty referenced shard: {name}")
        if validate_safetensors:
            try:
                entry["tensor_keys"] = _validate_safetensors_header(path)
            except (OSError, ValueError) as exc:
                issues.append(f"invalid safetensors shard {name}: {exc}")
        if hash_shards:
            entry["sha256"] = sha256_file(path)
        shards.append(entry)

    minimum_bytes = int(minimum_weight_gib * 1024**3)
    if total_weight_bytes < minimum_bytes:
        issues.append(
            f"weight size below minimum: found {total_weight_bytes / 1024**3:.3f} GiB, "
            f"required {minimum_weight_gib:.3f} GiB"
        )

    identity_files: dict[str, dict[str, Any]] = {}
    identity_hashes: dict[str, str] = {}
    for name in TOKENIZER_AND_PROCESSOR_FILES:
        path = root / name
        if not path.is_file():
            continue
        digest = sha256_file(path)
        identity_files[name] = {"bytes": path.stat().st_size, "sha256": digest}
        identity_hashes[name] = digest
    if "tokenizer.json" not in identity_files:
        issues.append("missing tokenizer.json")
    if not {"preprocessor_config.json", "processor_config.json"} & identity_files.keys():
        issues.append("missing processor/preprocessor config")

    temporary_files = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(marker in path.name.lower() for marker in (".incomplete", ".partial", ".tmp"))
    )
    if temporary_files:
        issues.append(f"download temporary files remain: {temporary_files[:3]}")

    return {
        "status": "passed" if not issues else "failed",
        "gate": "OPD_MODEL_SNAPSHOT_AUDIT_PASS" if not issues else "OPD_MODEL_SNAPSHOT_AUDIT_FAIL",
        "model_dir": str(root),
        "model": {
            "model_type": model_type,
            "architectures": architectures,
            "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
            "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        },
        "weights": {
            "referenced_parameter_keys": len(weight_map),
            "referenced_shards": len(referenced_shards),
            "total_bytes": total_weight_bytes,
            "total_gib": round(total_weight_bytes / 1024**3, 3),
            "minimum_gib": minimum_weight_gib,
            "headers_validated": validate_safetensors,
            "shards_hashed": hash_shards,
            "shards": shards,
        },
        "tokenizer_processor": {
            "files": identity_files,
            "fingerprint_sha256": _combined_fingerprint(identity_hashes) if identity_hashes else None,
        },
        "temporary_files": temporary_files,
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a downloaded OPD teacher/student model snapshot without GPU.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--minimum-weight-gib", type=float, default=0.0)
    parser.add_argument("--expected-model-type")
    parser.add_argument("--validate-safetensors", action="store_true")
    parser.add_argument("--hash-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Model audit already exists; use --force only after auditing it: {output}")
    report = audit_model_snapshot(
        args.model_dir,
        expected_shards=args.expected_shards,
        minimum_weight_gib=args.minimum_weight_gib,
        expected_model_type=args.expected_model_type,
        validate_safetensors=args.validate_safetensors,
        hash_shards=args.hash_shards,
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
