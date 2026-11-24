#!/usr/bin/env python3
"""Run the fixed-pool SDE training loop with a model-specific AVLLM runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sde_cctdh.config import load_config
from sde_cctdh.artifacts import write_jsonl
from sde_cctdh.runtime import (
    BatchCycle,
    append_jsonl,
    load_or_build_basis,
    load_runtime,
    parse_runtime_args,
    read_table,
)
from sde_cctdh.sde import (
    SDEWeights,
    apply_sde_step,
    build_sde_pools,
    compute_sde_objective,
)


def run(args: argparse.Namespace, runtime: Any | None = None) -> dict[str, Any]:
    config = load_config(args.config, require_resolved=args.require_resolved)
    stage = config["sde"]
    examples = read_table(args.examples)
    responses = read_table(args.responses)
    repair, protect, anchor = build_sde_pools(examples, responses)
    runtime = runtime or load_runtime(
        args.runtime,
        stage="sde",
        config=config,
        runtime_args=parse_runtime_args(args.runtime_arg),
    )
    parameters = list(runtime.trainable_parameters())
    basis = load_or_build_basis(
        path=args.retention_basis,
        runtime=runtime,
        rows=anchor,
        target_field="target_response",
        batch_size=args.basis_batch_size,
        rank=int(stage["retention_basis"]["rank"]),
        seed=int(stage["retention_basis"]["seed"]),
        parameters=parameters,
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pool_dir = output / "pools"
    write_jsonl(pool_dir / "repair.jsonl", repair)
    write_jsonl(pool_dir / "protect.jsonl", protect)
    write_jsonl(pool_dir / "anchor.jsonl", anchor)
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
    weights = SDEWeights(
        repair=float(stage["loss_weights"]["repair"]),
        protect=float(stage["loss_weights"]["protect"]),
        anchor=float(stage["loss_weights"]["anchor"]),
    )
    steps = int(args.steps or stage["steps"])
    last: dict[str, float] = {}
    exposure = {"repair": [], "protect": [], "anchor": []}
    for step in range(1, steps + 1):
        repair_rows = batches["repair"].next()
        protect_rows = batches["protect"].next()
        anchor_rows = batches["anchor"].next()
        repair_nll = runtime.response_nll(
            repair_rows, "target_response"
        )
        protect_nll = runtime.response_nll(
            protect_rows, "target_response"
        )
        anchor_nll = runtime.response_nll(
            anchor_rows, "target_response"
        )
        for name, rows in (
            ("repair", repair_rows),
            ("protect", protect_rows),
            ("anchor", anchor_rows),
        ):
            exposure[name].extend(str(row["qa_id"]) for row in rows)
        objective = compute_sde_objective(
            repair_nll,
            protect_nll,
            anchor_nll,
            weights=weights,
        )
        gradient = apply_sde_step(
            objective["loss"],
            parameters,
            runtime.optimizer,
            basis,
            clip_norm=args.gradient_clip_norm,
        )
        last = {
            "step": step,
            **{
                name: float(value.detach().cpu())
                for name, value in objective.items()
            },
            **gradient,
            "repair_ids": [str(row["qa_id"]) for row in repair_rows],
            "protect_ids": [str(row["qa_id"]) for row in protect_rows],
            "anchor_ids": [str(row["qa_id"]) for row in anchor_rows],
        }
        append_jsonl(log_path, last)
        print(json.dumps(last), flush=True)
    runtime.save(output / "adapter")
    torch.save({"basis": basis.detach().cpu()}, output / "sde_retention_basis.pt")
    summary = {
        "stage": "sde",
        "steps": steps,
        "repair_pool": len(repair),
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
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--runtime", required=True, help="MODULE:FACTORY")
    parser.add_argument("--runtime-arg", action="append", default=[])
    parser.add_argument("--retention-basis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
