from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from econochart.config import ROOT

LEGACY_TARGETS = (
    Path("data/raw/business_data.json"),
    Path("data/processed/sft_train.json"),
    Path("data/processed/dpo_train.json"),
    Path("data/processed/charts"),
)


def _validated_target(relative_path: Path) -> Path:
    data_root = (ROOT / "data").resolve()
    target = (ROOT / relative_path).resolve()
    if target == data_root or data_root not in target.parents:
        raise ValueError(f"Refusing legacy cleanup outside a named child of {data_root}: {target}")
    return target


def cleanup(*, apply: bool = False) -> list[dict[str, str | bool]]:
    results = []
    for relative_path in LEGACY_TARGETS:
        target = _validated_target(relative_path)
        exists = target.exists()
        kind = "directory" if target.is_dir() else "file"
        results.append(
            {
                "path": target.relative_to(ROOT).as_posix(),
                "exists": exists,
                "kind": kind,
                "action": "delete" if apply and exists else "would_delete" if exists else "skip_missing",
            }
        )
        if not apply or not exists:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    if apply:
        for relative_dir in (Path("data/raw"), Path("data/processed")):
            directory = _validated_target(relative_dir)
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove only the audited EconoChart-v1 generated files that are ignored by Git."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the exact legacy targets. Without this flag the command is a dry run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for result in cleanup(apply=args.apply):
        print(f"{result['action']}: {result['path']} ({result['kind']})")
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the exact paths above.")


if __name__ == "__main__":
    main()
