#!/usr/bin/env python3
"""Build MiniCPM-GPTQ artifacts around the shared SDE -> CC-TDH pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_pipeline as common  # noqa: E402
from sde_cctdh.artifacts import build_response_cache, write_jsonl  # noqa: E402
from sde_cctdh.config import load_config  # noqa: E402
from sde_cctdh.minicpm import MiniCPMRuntime  # noqa: E402
from sde_cctdh.runtime import read_table  # noqa: E402


# These stages contain no model-family-specific logic and are intentionally reused.
pairs = common.pairs
cctdh_pools = common.cctdh_pools
mine_parent_pools = common.mine_parent_pools
fit_teacher = common.fit_teacher
evaluate = common.evaluate


def _gold(frame: pd.DataFrame) -> pd.DataFrame:
    return common._gold(frame)


def _merge_labels(manifest: pd.DataFrame, labels: Path | None) -> pd.DataFrame:
    return common._merge_labels(manifest, labels)


def _runtime(args: argparse.Namespace) -> MiniCPMRuntime:
    config = load_config(args.config, require_resolved=True)
    runtime = MiniCPMRuntime(
        stage="sde",
        config=config,
        model_path=args.model_path,
        protocol_config=args.protocol_config,
        protocol_name=args.protocol_name,
        dtype=args.dtype,
        attention=args.attention,
        device=args.device,
    )
    if getattr(args, "sde_adapter", None) is not None:
        runtime.load_sde_for_inference(args.sde_adapter)
    if getattr(args, "cctdh_adapter", None) is not None:
        runtime.load_cctdh_for_inference(args.cctdh_adapter)
    return runtime


def decode(args: argparse.Namespace) -> dict[str, Any]:
    rows = _merge_labels(read_table(args.manifest), args.labels)
    required = {"qa_id", "source_id", "media_path", "gold_answer"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"decode manifest missing columns: {sorted(missing)}")
    runtime = _runtime(args)
    try:
        predictions = pd.DataFrame(runtime.generate(rows.to_dict("records")))
    finally:
        runtime.close()
    joined = rows.merge(
        predictions,
        on=["qa_id", "source_id"],
        how="left",
        validate="one_to_one",
    )
    prefix = (
        "method"
        if args.cctdh_adapter is not None
        else "parent"
        if args.sde_adapter is not None
        else "base"
    )
    joined = joined.rename(
        columns={
            "raw_response": f"{prefix}_response",
            "answer": f"{prefix}_answer",
        }
    )
    write_jsonl(args.output, joined)
    summary: dict[str, Any] = {
        "stage": f"decode_{prefix}",
        "rows": len(joined),
        "invalid": int(joined[f"{prefix}_answer"].eq("__invalid__").sum()),
        "output": str(args.output),
    }
    if prefix == "base":
        cache = build_response_cache(
            rows,
            joined[["qa_id", "base_answer", "base_response"]],
        )
        response_cache = args.response_cache or args.output.with_name(
            "base_response_cache.jsonl"
        )
        write_jsonl(response_cache, cache)
        summary["response_cache"] = str(response_cache)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def extract_hidden(args: argparse.Namespace) -> dict[str, Any]:
    rows = _merge_labels(read_table(args.manifest), args.labels)
    runtime = _runtime(args)
    try:
        hidden = pd.DataFrame(
            runtime.extract_hidden(
                rows.to_dict("records"),
                layer=args.hidden_layer,
                target_field=args.target_field,
            )
        )
    finally:
        runtime.close()
    write_jsonl(args.output, hidden)
    return {
        "stage": "extract_parent_hidden",
        "rows": len(hidden),
        "hidden_layer": args.hidden_layer,
        "output": str(args.output),
    }


def export_deployed(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _runtime(args)
    try:
        runtime.save_deployed(args.output)
    finally:
        runtime.close()
    return {
        "stage": "export_deployed",
        "output": str(args.output),
        "load_order": ["sde_adapter.pt", "cctdh_adapter.pt"],
        "experts_required_at_inference": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = common.build_parser()
    commands = parser._subparsers._group_actions[0].choices
    commands["decode"].set_defaults(function=decode)
    commands["extract-parent-hidden"].set_defaults(function=extract_hidden)
    commands["extract-parent-hidden"].set_defaults(hidden_layer=34)
    commands["fit-teacher"].set_defaults(hidden_layer=34)
    commands["export-deployed"].set_defaults(function=export_deployed)
    for name in ("decode", "extract-parent-hidden", "export-deployed"):
        command = commands[name]
        for action in command._actions:
            if action.dest == "device" and action.default == "cuda":
                action.default = "auto"
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(arguments.function(arguments), indent=2))
