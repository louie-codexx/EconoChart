from __future__ import annotations

import argparse
import json
from typing import Any

from econochart.config import load_config
from econochart.distillation.qualification import run_paired_gate


def qualify_round(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Apply the frozen development gate after an OPD student update."""
    return run_paired_gate(
        config,
        pass_gate="OPD_ROUND1_DEVELOPMENT_GATE_PASS",
        fail_gate="OPD_ROUND1_DEVELOPMENT_GATE_FAIL",
        context="OPD round-1 development gate",
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the frozen GRPO baseline and an OPD round candidate on development data."
    )
    parser.add_argument("--config", default="configs/opd/round1_development_gate_v1.yaml")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = qualify_round(load_config(args.config), force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
