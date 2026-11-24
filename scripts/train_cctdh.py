#!/usr/bin/env python3
"""Run CC-TDH residual-LoRA training with a model-specific AVLLM runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sde_cctdh.cctdh import CCTDHConfig, apply_cctdh_step
from sde_cctdh.artifacts import write_jsonl
from sde_cctdh.config import load_config
from sde_cctdh.runtime import (
    BatchCycle,
    append_jsonl,
    load_or_build_basis,
    load_runtime,
    parse_runtime_args,
    read_table,
)


def _mean(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.numel() == 0:
        raise ValueError(f"{name} runtime output is empty")
    return value.mean() if value.ndim else value


def run(args: argparse.Namespace, runtime: Any | None = None) -> dict[str, Any]:
    config = load_config(args.config, require_resolved=args.require_resolved)
    stage = config["cctdh"]
    repair = read_table(args.validated_repair)
    protect = read_table(args.protect)
    anchor = read_table(args.anchor)
    required = (
        (repair, args.repair_target_field),
        (protect, args.retention_target_field),
        (anchor, args.retention_target_field),
    )
    for frame, field in required:
        if frame.empty or field not in frame.columns:
            raise ValueError(f"non-empty training pool must contain {field!r}")
    runtime = runtime or load_runtime(
        args.runtime,
        stage="cctdh",
        config=config,
        runtime_args=parse_runtime_args(args.runtime_arg),
    )
    if not hasattr(runtime, "detail_loss"):
        raise TypeError("CC-TDH runtime must implement detail_loss(rows)")
    parameters = list(runtime.trainable_parameters())
    retention = pd.concat([protect, anchor], ignore_index=True)
    basis = load_or_build_basis(
        path=args.retention_basis,
        runtime=runtime,
        rows=retention,
        target_field=args.retention_target_field,
        batch_size=args.basis_batch_size,
        rank=int(stage["retention_basis"]["rank"]),
        seed=int(stage["retention_basis"]["seed"]),
        parameters=parameters,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pool_dir = output / "pools"
    write_jsonl(pool_dir / "validated_repair.jsonl", repair)
    write_jsonl(pool_dir / "parent_protect.jsonl", protect)
    write_jsonl(pool_dir / "parent_anchor.jsonl", anchor)
    log_path = output / "train_log.jsonl"
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(log_path)
    log_path.unlink(missing_ok=True)
    seed = int(args.seed)
    batches = {
        "repair": BatchCycle(repair, args.batch_size, seed),
        "protect": BatchCycle(protect, args.batch_size, seed + 1),
        "anchor": BatchCycle(anchor, args.batch_size, seed + 2),
    }
    weights = stage["loss_weights"]
    method_config = CCTDHConfig(
        protect_weight=float(weights["protect"]),
        anchor_weight=float(weights["anchor"]),
        detail_max_weight=float(weights["detail_max"]),
        detail_warmup_steps=int(stage["detail_warmup_steps"]),
        norm_cap_ratio=float(stage["detail_norm_cap_ratio"]),
        gradient_clip_norm=(
            args.gradient_clip_norm
            if args.gradient_clip_norm is not None
            else stage.get("optimizer", {}).get("gradient_clip_norm")
        ),
    )
    steps = int(args.steps or stage["steps"])
    last: dict[str, Any] = {}
    exposure = {"repair": [], "protect": [], "anchor": []}
    for step in range(1, steps + 1):
        repair_rows = batches["repair"].next()
        protect_rows = batches["protect"].next()
        anchor_rows = batches["anchor"].next()
        for name, rows in (
            ("repair", repair_rows),
            ("protect", protect_rows),
            ("anchor", anchor_rows),
        ):
            exposure[name].extend(str(row["qa_id"]) for row in rows)
        repair_loss = _mean(
            runtime.response_nll(repair_rows, args.repair_target_field),
            "repair",
        )
        protect_loss = _mean(
            runtime.response_nll(
                protect_rows, args.retention_target_field
            ),
            "protect",
        )
        anchor_loss = _mean(
            runtime.response_nll(
                anchor_rows, args.retention_target_field
            ),
            "anchor",
        )
        detail_loss = _mean(runtime.detail_loss(repair_rows), "detail")
        result = apply_cctdh_step(
            repair_gold_loss=repair_loss,
            protect_parent_loss=protect_loss,
            anchor_parent_loss=anchor_loss,
            detail_loss=detail_loss,
            parameters=parameters,
            optimizer=runtime.optimizer,
            parent_retention_basis=basis,
            residual_step=step,
            config=method_config,
        )
        last = {
            "step": step,
            "repair_loss": float(repair_loss.detach().cpu()),
            "protect_loss": float(protect_loss.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "detail_loss": float(detail_loss.detach().cpu()),
            "main_gradient_norm": float(result.main.norm()),
            "retention_gradient_norm": float(result.retention.norm()),
            "detail_gradient_norm": float(result.detail_raw.norm()),
            "final_gradient_norm": float(result.final.norm()),
            "detail_weight": result.detail_weight,
            "conflict_triggered": result.conflict_triggered,
            "norm_cap_triggered": result.norm_cap_triggered,
            "fallback": result.fallback,
            "projection_removed_ratio": result.projection_removed_ratio,
            "gradient_clip_scale": result.gradient_clip_scale,
            "repair_ids": [str(row["qa_id"]) for row in repair_rows],
            "protect_ids": [str(row["qa_id"]) for row in protect_rows],
            "anchor_ids": [str(row["qa_id"]) for row in anchor_rows],
        }
        append_jsonl(log_path, last)
        print(json.dumps(last), flush=True)
    runtime.save(output / "adapter")
    torch.save({"basis": basis.detach().cpu()}, output / "parent_retention_basis.pt")
    summary = {
        "stage": "cctdh",
        "steps": steps,
        "validated_repair_pool": len(repair),
        "protect_pool": len(protect),
        "anchor_pool": len(anchor),
        "basis_rank": int(basis.shape[1]),
        "exposure": {
            name: {
                "count": len(values),
                "unique": len(set(values)),
                "qa_ids": values,
            }
            for name, values in exposure.items()
        },
        "last": last,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validated-repair", type=Path, required=True)
    parser.add_argument("--protect", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--runtime", required=True, help="MODULE:FACTORY")
    parser.add_argument("--runtime-arg", action="append", default=[])
    parser.add_argument("--retention-basis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-target-field", default="gold_response")
    parser.add_argument(
        "--retention-target-field", default="retention_target_response"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--basis-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--require-resolved", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
