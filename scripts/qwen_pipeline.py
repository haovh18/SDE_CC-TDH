#!/usr/bin/env python3
"""Build reproducible Qwen artifacts around the SDE -> CC-TDH stages."""
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

from sde_cctdh.artifacts import (  # noqa: E402
    build_expert_pair_input,
    build_response_cache,
    fit_teacher_bundle,
    prepare_cctdh_artifacts,
    write_jsonl,
)
from sde_cctdh.cctdh import (  # noqa: E402
    build_parent_retention_pools,
    mine_residual_repairs,
)
from sde_cctdh.config import load_config  # noqa: E402
from sde_cctdh.metrics import evaluate_predictions, source_bootstrap  # noqa: E402
from sde_cctdh.qwen import QwenOmniRuntime  # noqa: E402
from sde_cctdh.runtime import read_table  # noqa: E402


def _gold(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "gold_answer" not in out and "gold_index" in out:
        out["gold_answer"] = out["gold_index"].map(lambda value: "ABCD"[int(value)])
    return out


def _merge_labels(manifest: pd.DataFrame, labels: Path | None) -> pd.DataFrame:
    frame = manifest.copy()
    if labels is not None:
        label_frame = _gold(read_table(labels))
        columns = [
            column
            for column in ("qa_id", "gold_answer", "gold_response")
            if column in label_frame
        ]
        frame = frame.merge(
            label_frame[columns],
            on="qa_id",
            how="left",
            validate="one_to_one",
        )
    return _gold(frame)


def _runtime(args: argparse.Namespace) -> QwenOmniRuntime:
    config = load_config(args.config, require_resolved=False)
    runtime = QwenOmniRuntime(
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
    if required - set(rows.columns):
        raise ValueError(f"decode manifest missing columns: {sorted(required - set(rows.columns))}")
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


def pairs(args: argparse.Namespace) -> dict[str, Any]:
    expert_predictions = pd.concat(
        [read_table(path) for path in args.expert_predictions],
        ignore_index=True,
    )
    result = build_expert_pair_input(
        read_table(args.base_outputs),
        expert_predictions,
        read_table(args.metadata) if args.metadata else None,
    )
    write_jsonl(args.output, result)
    return {
        "stage": "expert_pairs",
        "rows": len(result),
        "qa": int(result["qa_id"].nunique()),
        "experts": sorted(result["expert_name"].astype(str).unique()),
    }


def cctdh_pools(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, require_resolved=False)
    artifacts = prepare_cctdh_artifacts(
        sde_repair_pool=read_table(args.sde_repair_pool),
        example_manifest=read_table(args.examples),
        pair_audit=read_table(args.pair_audit),
        parent_outputs=read_table(args.parent_outputs),
        intervention_effects=read_table(args.intervention_effects),
        detail_targets=(
            read_table(args.detail_targets)
            if args.detail_targets is not None
            else None
        ),
        local_details=(
            read_table(args.local_details)
            if args.local_details is not None
            else None
        ),
        latent_dim=int(config["teacher"]["latent_dim"]),
        n_folds=int(config["routing"]["source_folds"]),
        seed=int(config["routing"]["seed"]),
        tie_order=config["routing"]["tie_order"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            write_jsonl(args.output / f"{name}.jsonl", value)
            counts[name] = len(value)
        else:
            torch.save(value, args.output / f"{name}.pt")
            counts[name] = len(value)
    summary = {
        "stage": "prepare_cctdh",
        "counts": counts,
        "default_parent": "sde",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def mine_parent_pools(args: argparse.Namespace) -> dict[str, Any]:
    repair = mine_residual_repairs(
        read_table(args.sde_repair_pool),
        read_table(args.parent_outputs),
        read_table(args.pair_audit),
    )
    protect, anchor = build_parent_retention_pools(
        read_table(args.examples),
        read_table(args.parent_outputs),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "residual_repair": repair,
        "parent_protect": protect,
        "parent_anchor": anchor,
    }
    for name, frame in outputs.items():
        write_jsonl(args.output / f"{name}.jsonl", frame)
    return {
        "stage": "mine_parent_pools",
        "counts": {name: len(frame) for name, frame in outputs.items()},
    }


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


def fit_teacher(args: argparse.Namespace) -> dict[str, Any]:
    fit = read_table(args.teacher_fit)
    hidden = read_table(args.parent_hidden)
    fit = fit.merge(
        hidden[["qa_id", "source_id", "parent_hidden"]],
        on=["qa_id", "source_id"],
        how="left",
        validate="one_to_one",
    )
    bundle = fit_teacher_bundle(
        fit,
        read_table(args.supervision),
        hidden_layer=args.hidden_layer,
        ridge=args.ridge,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    manifest = {
        "stage": "fit_teacher",
        "hidden_layer": args.hidden_layer,
        "answer_position": "response_start_minus_one",
        "ridge": args.ridge,
        "experts": sorted(bundle["teachers"]),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    base = read_table(args.base_outputs)
    method = read_table(args.method_outputs)
    examples = read_table(args.examples)
    required_base = {"qa_id", "base_answer"}
    required_method = {"qa_id", "method_answer"}
    if required_base - set(base.columns) or required_method - set(method.columns):
        raise ValueError("prediction outputs do not have Base/method answer columns")
    columns = [
        column
        for column in ("qa_id", "source_id", "gold_answer", "example_state")
        if column in examples
    ]
    rows = examples[columns].merge(
        base[["qa_id", "base_answer"]],
        on="qa_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        method[["qa_id", "method_answer"]],
        on="qa_id",
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != len(examples):
        raise ValueError("Base/method predictions do not cover the evaluation manifest")
    metrics = evaluate_predictions(rows)
    bootstrap = {
        metric: source_bootstrap(
            rows,
            metric=metric,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
        for metric in args.bootstrap_metric
    }
    result = {
        "stage": "evaluate_sde_cctdh",
        "rows": len(rows),
        "metrics": metrics,
        "source_bootstrap": bootstrap,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.per_qa is not None:
        write_jsonl(args.per_qa, rows)
    return result


def export_deployed(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _runtime(args)
    try:
        runtime.save_deployed(args.output)
    finally:
        runtime.close()
    return {
        "stage": "export_deployed",
        "output": str(args.output),
        "experts_required_at_inference": False,
    }


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=ROOT / "baseline/config.yaml",
    )
    parser.add_argument("--protocol-name", default="music_avqa_42")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    decode_parser = commands.add_parser("decode")
    _model_arguments(decode_parser)
    decode_parser.add_argument("--manifest", type=Path, required=True)
    decode_parser.add_argument("--labels", type=Path)
    decode_parser.add_argument("--sde-adapter", type=Path)
    decode_parser.add_argument("--cctdh-adapter", type=Path)
    decode_parser.add_argument("--response-cache", type=Path)
    decode_parser.add_argument("--output", type=Path, required=True)
    decode_parser.set_defaults(function=decode)

    pair_parser = commands.add_parser("expert-pairs")
    pair_parser.add_argument("--base-outputs", type=Path, required=True)
    pair_parser.add_argument(
        "--expert-predictions",
        type=Path,
        action="append",
        required=True,
    )
    pair_parser.add_argument("--metadata", type=Path)
    pair_parser.add_argument("--output", type=Path, required=True)
    pair_parser.set_defaults(function=pairs)

    pool_parser = commands.add_parser("prepare-cctdh")
    pool_parser.add_argument("--config", type=Path, required=True)
    pool_parser.add_argument("--sde-repair-pool", type=Path, required=True)
    pool_parser.add_argument("--examples", type=Path, required=True)
    pool_parser.add_argument("--pair-audit", type=Path, required=True)
    pool_parser.add_argument("--parent-outputs", type=Path, required=True)
    pool_parser.add_argument("--intervention-effects", type=Path, required=True)
    detail_source = pool_parser.add_mutually_exclusive_group(required=True)
    detail_source.add_argument("--detail-targets", type=Path)
    detail_source.add_argument("--local-details", type=Path)
    pool_parser.add_argument("--output", type=Path, required=True)
    pool_parser.set_defaults(function=cctdh_pools)

    mine_parser = commands.add_parser("mine-parent-pools")
    mine_parser.add_argument("--sde-repair-pool", type=Path, required=True)
    mine_parser.add_argument("--examples", type=Path, required=True)
    mine_parser.add_argument("--pair-audit", type=Path, required=True)
    mine_parser.add_argument("--parent-outputs", type=Path, required=True)
    mine_parser.add_argument("--output", type=Path, required=True)
    mine_parser.set_defaults(function=mine_parent_pools)

    hidden_parser = commands.add_parser("extract-parent-hidden")
    _model_arguments(hidden_parser)
    hidden_parser.add_argument("--manifest", type=Path, required=True)
    hidden_parser.add_argument("--labels", type=Path)
    hidden_parser.add_argument("--sde-adapter", type=Path, required=True)
    hidden_parser.add_argument("--hidden-layer", type=int, default=30)
    hidden_parser.add_argument("--target-field", default="gold_response")
    hidden_parser.add_argument("--output", type=Path, required=True)
    hidden_parser.set_defaults(function=extract_hidden)

    teacher_parser = commands.add_parser("fit-teacher")
    teacher_parser.add_argument("--teacher-fit", type=Path, required=True)
    teacher_parser.add_argument("--parent-hidden", type=Path, required=True)
    teacher_parser.add_argument("--supervision", type=Path, required=True)
    teacher_parser.add_argument("--hidden-layer", type=int, default=30)
    teacher_parser.add_argument("--ridge", type=float, default=1.0)
    teacher_parser.add_argument("--output", type=Path, required=True)
    teacher_parser.set_defaults(function=fit_teacher)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--base-outputs", type=Path, required=True)
    evaluate_parser.add_argument("--method-outputs", type=Path, required=True)
    evaluate_parser.add_argument("--examples", type=Path, required=True)
    evaluate_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    evaluate_parser.add_argument(
        "--bootstrap-metric",
        action="append",
        default=["Accuracy", "BER", "DR", "RR", "PR", "UFR"],
    )
    evaluate_parser.add_argument("--seed", type=int, default=18)
    evaluate_parser.add_argument("--per-qa", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(function=evaluate)

    export_parser = commands.add_parser("export-deployed")
    _model_arguments(export_parser)
    export_parser.add_argument("--sde-adapter", type=Path, required=True)
    export_parser.add_argument("--cctdh-adapter", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.set_defaults(function=export_deployed)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(arguments.function(arguments), indent=2))
