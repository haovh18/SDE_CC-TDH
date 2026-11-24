#!/usr/bin/env python3
"""Prepare, train, and decode the three expert-transfer paper baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from baseline.protocol import read_jsonl, write_jsonl  # noqa: E402
from sde_cctdh.config import load_config as load_method_config  # noqa: E402
from sde_cctdh.qwen import (  # noqa: E402
    QwenOmniRuntime,
    QwenProtocol,
    adapter_state,
    load_adapter_state,
)
from sde_cctdh.runtime import BatchCycle  # noqa: E402


METHODS = ("vanilla_kd", "rwkd", "ufa")
EXPERT_ORDER = ("imagebind", "muq", "videomae")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("named paths must be unique NAME=PATH values")
        result[name] = Path(path)
    return result


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _fold(source_id: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}::rwkd::{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _temperature_scale_teachers(
    teacher_probabilities: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    teachers = teacher_probabilities.float().clamp_min(1e-12)
    teachers = teachers / teachers.sum(dim=1, keepdim=True)
    teachers = teachers.pow(1.0 / temperature)
    return teachers / teachers.sum(dim=1, keepdim=True)


def kd_loss(
    student_scores: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    temperature: float,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Complete-answer Vanilla KD or non-negative reliability-weighted KD."""
    if student_scores.ndim != 1 or teacher_probabilities.ndim != 2:
        raise ValueError("expected scores [class] and teachers [expert, class]")
    if teacher_probabilities.shape[1] != student_scores.numel():
        raise ValueError("teacher and student class spaces differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teachers = _temperature_scale_teachers(teacher_probabilities, temperature)
    log_student = F.log_softmax(student_scores.float() / temperature, dim=0)
    if reliability is None:
        value = F.kl_div(log_student, teachers.mean(dim=0), reduction="sum")
    else:
        losses = torch.stack(
            [
                F.kl_div(log_student, teacher, reduction="sum")
                for teacher in teachers
            ]
        )
        weights = reliability.float()
        if weights.shape != losses.shape or bool((weights < 0).any()):
            raise ValueError("RWKD weights must be non-negative per-expert values")
        value = (weights * losses).mean()
    return value * temperature**2


def kd_score_gradient(
    student_scores: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    temperature: float,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Analytical d(KD)/d(complete-answer score), used for low-memory replay."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teachers = _temperature_scale_teachers(teacher_probabilities, temperature)
    student = torch.softmax(student_scores.float() / temperature, dim=0)
    if reliability is None:
        return temperature * (student - teachers.mean(dim=0))
    weights = reliability.float()
    if bool((weights < 0).any()):
        raise ValueError("RWKD weights must be non-negative")
    return temperature * (
        weights.mean() * student - (weights[:, None] * teachers).mean(dim=0)
    )


class UFAProjectors(nn.Module):
    def __init__(self, hidden_dim: int, expert_dims: dict[str, int], rank: int):
        super().__init__()
        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dimension, rank, bias=False),
                    nn.Linear(rank, hidden_dim, bias=False),
                )
                for name, dimension in expert_dims.items()
            }
        )


def unsigned_alignment_loss(
    hidden: torch.Tensor,
    embeddings: dict[str, torch.Tensor],
    available: dict[str, torch.Tensor],
    projectors: UFAProjectors,
) -> torch.Tensor:
    """Mean positive cosine alignment over available example-expert pairs."""
    terms: list[torch.Tensor] = []
    pair_count = 0
    for expert, values in embeddings.items():
        chosen = torch.nonzero(available[expert], as_tuple=False).flatten()
        if chosen.numel() == 0:
            continue
        student = hidden[chosen].float()
        projected_expert = projectors.experts[expert](values[chosen].float())
        terms.append(
            1.0 - F.cosine_similarity(projected_expert, student, dim=-1)
        )
        pair_count += int(chosen.numel())
    if not terms:
        raise ValueError("UFA batch has no available expert pairs")
    return torch.cat(terms).sum() / pair_count


def _load_probability_table(
    path: Path,
    expert_manifest: pd.DataFrame,
    default_classes: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
            row_indices = np.asarray(payload["row_indices"], dtype=np.int64)
            classes = list(map(str, payload["vocab"].tolist()))
        qa_ids = expert_manifest.iloc[row_indices]["qa_id"].astype(str).tolist()
        return dict(zip(qa_ids, probabilities, strict=True)), classes
    rows = read_jsonl(path)
    table = {
        str(row["qa_id"]): np.asarray(row["probabilities"], dtype=np.float32)
        for row in rows
    }
    return table, list(map(str, default_classes))


def _crossfit_weights(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    expert_names: Sequence[str],
    classes: Sequence[str],
    *,
    seed: int,
    folds: int,
) -> list[dict[str, Any]]:
    fold_ids = frame["source_id"].astype(str).map(lambda value: _fold(value, seed, folds))
    gold = frame["gold_response"].astype(str).to_numpy()
    pending = []
    raw_values = []
    for expert_index, expert in enumerate(expert_names):
        values = probabilities[:, expert_index]
        predictions = np.asarray(classes)[values.argmax(axis=1)]
        correct = predictions == gold
        sorted_values = np.sort(values, axis=1)
        gaps = sorted_values[:, -1] - sorted_values[:, -2]
        for heldout_fold in range(folds):
            fit = fold_ids.ne(heldout_fold).to_numpy()
            held = fold_ids.eq(heldout_fold).to_numpy()
            if not fit.any() or not held.any():
                raise ValueError("RWKD cross-fitting produced an empty fold")
            quantiles = np.quantile(gaps[fit], np.linspace(0, 1, 11))
            quantiles[0], quantiles[-1] = -np.inf, np.inf
            calibration = []
            for bin_index in range(10):
                selected = fit & (gaps > quantiles[bin_index]) & (
                    gaps <= quantiles[bin_index + 1]
                )
                count = int(selected.sum())
                calibration.append((int(correct[selected].sum()) + 1) / (count + 2))
            groups = frame["question_group"].astype(str)
            group_stats = {}
            for group in sorted(set(groups[fit])):
                selected = fit & groups.eq(group).to_numpy()
                group_stats[group] = (
                    int(correct[selected].sum()) + 1
                ) / (int(selected.sum()) + 2)
            fallback = (int(correct[fit].sum()) + 1) / (int(fit.sum()) + 2)
            for index in np.flatnonzero(held):
                bin_index = int(np.searchsorted(quantiles, gaps[index], side="left") - 1)
                bin_index = max(0, min(9, bin_index))
                rho = float(group_stats.get(str(groups.iloc[index]), fallback))
                confidence = float(calibration[bin_index])
                raw = rho * confidence
                pending.append(
                    {
                        "qa_id": str(frame.iloc[index]["qa_id"]),
                        "expert": expert,
                        "fold": heldout_fold,
                        "rho": rho,
                        "calibrated_confidence": confidence,
                        "raw_weight": raw,
                    }
                )
                raw_values.append(raw)
    normalizer = float(np.mean(raw_values))
    for row in pending:
        row["r_bar"] = float(np.clip(row["raw_weight"] / normalizer, 0.25, 2.0))
    return pending


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    _seed(args.seed)
    protocol = QwenProtocol.from_yaml(args.protocol_config, args.protocol_name)
    inference = pd.DataFrame(read_jsonl(args.manifest))
    labels = pd.DataFrame(read_jsonl(args.labels))
    frame = inference.merge(
        labels.drop(columns=["dataset", "source_id"], errors="ignore"),
        on="qa_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(inference):
        raise ValueError("every distillation example requires one label")
    frame["qa_id"] = frame["qa_id"].astype(str)
    frame["source_id"] = frame["source_id"].astype(str)
    if "question_group" not in frame:
        frame["question_group"] = "all"
    else:
        frame["question_group"] = frame["question_group"].fillna("all")
    frame["gold_response"] = [
        protocol.gold_response(row) for row in frame.to_dict(orient="records")
    ]
    expert_manifest = pd.read_json(args.expert_manifest, lines=True)
    expert_manifest["qa_id"] = expert_manifest["qa_id"].astype(str)
    probability_paths = _named_paths(args.expert_probabilities)
    if tuple(probability_paths) != EXPERT_ORDER:
        raise ValueError(f"expert probability order must be {EXPERT_ORDER}")
    probability_tables = {}
    classes = None
    for expert, path in probability_paths.items():
        table, current_classes = _load_probability_table(
            path,
            expert_manifest,
            protocol.answer_vocabulary,
        )
        if classes is None:
            classes = current_classes
        elif current_classes != classes:
            raise ValueError("expert candidate answer spaces differ")
        probability_tables[expert] = table
    assert classes is not None
    if list(protocol.answer_vocabulary) != classes:
        raise ValueError("expert and AVLLM candidate answer spaces differ")
    probabilities = np.zeros(
        (len(frame), len(EXPERT_ORDER), len(classes)), dtype=np.float32
    )
    for row_index, qa_id in enumerate(frame["qa_id"]):
        for expert_index, expert in enumerate(EXPERT_ORDER):
            if qa_id not in probability_tables[expert]:
                raise ValueError(f"{expert}: no OOF distribution for {qa_id}")
            probabilities[row_index, expert_index] = probability_tables[expert][qa_id]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("expert probabilities must be finite and non-negative")
    if (probabilities.sum(axis=2) <= 0).any():
        raise ValueError("every expert distribution must have positive mass")
    probabilities /= probabilities.sum(axis=2, keepdims=True).clip(1e-12)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "training_manifest.jsonl", frame.to_dict(orient="records"))
    _write_npz(
        output / "teacher_distributions.npz",
        qa_ids=frame["qa_id"].to_numpy(),
        expert_names=np.asarray(EXPERT_ORDER),
        class_names=np.asarray(classes),
        probabilities=probabilities,
    )
    weights = _crossfit_weights(
        frame,
        probabilities,
        EXPERT_ORDER,
        classes,
        seed=args.seed,
        folds=args.folds,
    )
    write_jsonl(output / "rwkd_weights.jsonl", weights)

    embedding_paths = _named_paths(args.expert_arrays)
    if embedding_paths:
        if tuple(embedding_paths) != EXPERT_ORDER:
            raise ValueError(f"expert array order must be {EXPERT_ORDER}")
        manifest_index = {
            qa_id: index for index, qa_id in enumerate(expert_manifest["qa_id"])
        }
        embedding_payload: dict[str, Any] = {
            "qa_ids": frame["qa_id"].to_numpy(),
            "expert_names": np.asarray(EXPERT_ORDER),
        }
        for expert, path in embedding_paths.items():
            with np.load(path, allow_pickle=False) as payload:
                media = np.asarray(payload["media"], dtype=np.float32)
            indexes = [manifest_index[qa_id] for qa_id in frame["qa_id"]]
            selected_media = media[indexes]
            if not np.isfinite(selected_media).all():
                raise ValueError(f"{expert}: non-finite UFA embeddings")
            embedding_payload[f"{expert}_embeddings"] = selected_media
            embedding_payload[f"{expert}_available"] = np.ones(len(frame), bool)
        _write_npz(output / "expert_embeddings.npz", **embedding_payload)

    schedule_rows = []
    if args.sde_log is not None:
        for row in read_jsonl(args.sde_log):
            qa_ids = []
            for key in ("repair_ids", "protect_ids", "anchor_ids"):
                qa_ids.extend(map(str, row.get(key, [])))
            if not qa_ids:
                raise ValueError("SDE log contains an empty exposure step")
            schedule_rows.append({"step": int(row["step"]), "qa_ids": qa_ids})
        write_jsonl(output / "training_schedule.jsonl", schedule_rows)
    summary = {
        "stage": "prepare_distillation",
        "rows": len(frame),
        "experts": list(EXPERT_ORDER),
        "classes": len(classes),
        "source_folds": args.folds,
        "seed": args.seed,
        "embeddings": bool(embedding_paths),
        "scheduled_steps": len(schedule_rows),
        "rpu_used_by_loss": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _forward(
    runtime: QwenOmniRuntime,
    row: dict[str, Any],
    answer: str,
) -> tuple[torch.Tensor, dict[str, int]]:
    inputs, metadata = runtime.prepare_training(row, answer)
    output = runtime.model.thinker(
        **inputs,
        use_audio_in_video=True,
        use_cache=False,
        return_dict=True,
    )
    return output.loss, metadata


def _scheduled_batch(
    rows: pd.DataFrame,
    schedule: list[dict[str, Any]],
    step: int,
) -> list[dict[str, Any]]:
    by_id = rows.set_index("qa_id", drop=False)
    qa_ids = list(map(str, schedule[(step - 1) % len(schedule)]["qa_ids"]))
    missing = [qa_id for qa_id in qa_ids if qa_id not in by_id.index]
    if missing:
        raise ValueError(f"schedule references absent QA IDs: {missing[:3]}")
    return by_id.loc[qa_ids].to_dict(orient="records")


def train(args: argparse.Namespace) -> dict[str, Any]:
    baseline_config = yaml.safe_load(args.baseline_config.read_text(encoding="utf-8"))
    settings = baseline_config["distillation"][args.method]
    seed = int(args.seed if args.seed is not None else baseline_config["seed"])
    _seed(seed)
    method_config = load_method_config(args.method_config, require_resolved=False)
    runtime = QwenOmniRuntime(
        stage="sde",
        config=method_config,
        model_path=args.model_path,
        protocol_config=args.protocol_config,
        protocol_name=args.protocol_name,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
    )
    rows = pd.DataFrame(read_jsonl(args.bundle / "training_manifest.jsonl"))
    rows["qa_id"] = rows["qa_id"].astype(str)
    with np.load(args.bundle / "teacher_distributions.npz", allow_pickle=False) as data:
        teacher_qa = list(map(str, data["qa_ids"].tolist()))
        experts = list(map(str, data["expert_names"].tolist()))
        classes = list(map(str, data["class_names"].tolist()))
        teacher_probabilities = torch.from_numpy(data["probabilities"].astype(np.float32))
    teacher_index = {qa_id: index for index, qa_id in enumerate(teacher_qa)}
    if experts != list(EXPERT_ORDER):
        raise ValueError(f"teacher expert order must be {EXPERT_ORDER}")
    if classes != list(runtime.protocol.answer_vocabulary):
        raise ValueError("teacher and runtime answer spaces differ")
    if set(rows["qa_id"]) - set(teacher_index):
        raise ValueError("teacher bundle does not cover the training manifest")
    weights = {}
    if args.method == "rwkd":
        weights = {
            (str(row["qa_id"]), str(row["expert"])): float(row["r_bar"])
            for row in read_jsonl(args.bundle / "rwkd_weights.jsonl")
        }
    embedding_data = None
    embedding_index = None
    projectors = None
    hidden_capture: dict[str, torch.Tensor] = {}
    hook = None
    parameters = runtime.trainable_parameters()
    if args.method == "ufa":
        embedding_path = args.bundle / "expert_embeddings.npz"
        if not embedding_path.is_file():
            raise FileNotFoundError("UFA requires expert_embeddings.npz")
        embedding_data = np.load(embedding_path, allow_pickle=False)
        embedding_index = {
            str(qa_id): index
            for index, qa_id in enumerate(embedding_data["qa_ids"].tolist())
        }
        expert_dims = {
            expert: int(embedding_data[f"{expert}_embeddings"].shape[1])
            for expert in experts
        }
        hidden_dim = int(runtime.model.thinker.config.text_config.hidden_size)
        projectors = UFAProjectors(
            hidden_dim,
            expert_dims,
            rank=int(settings["projector_rank"]),
        ).to(runtime.device, torch.float32)
        parameters += list(projectors.parameters())

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden_capture["value"] = output[0] if isinstance(output, tuple) else output

        hook = runtime.model.thinker.model.layers[
            int(settings["hidden_layer"])
        ].register_forward_hook(capture)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    cycle = BatchCycle(rows, args.batch_size, seed)
    schedule_path = args.schedule or args.bundle / "training_schedule.jsonl"
    schedule = read_jsonl(schedule_path) if schedule_path.is_file() else []
    steps = int(args.steps if args.steps is not None else settings["steps"])
    temperature = None
    loss_weight = None
    if args.method in {"vanilla_kd", "rwkd"}:
        temperature = float(
            args.temperature
            if args.temperature is not None
            else settings["temperature"]
        )
    else:
        loss_weight = float(
            args.loss_weight if args.loss_weight is not None else settings["lambda"]
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = []
    try:
        for step in range(1, steps + 1):
            batch = _scheduled_batch(rows, schedule, step) if schedule else cycle.next()
            optimizer.zero_grad(set_to_none=True)
            supervised_values = []
            auxiliary_values = []
            if args.method in {"vanilla_kd", "rwkd"}:
                assert temperature is not None
                for row in batch:
                    with torch.no_grad():
                        scores = []
                        for answer in classes:
                            candidate_loss, _ = _forward(runtime, row, answer)
                            scores.append(-candidate_loss.detach().float())
                        score_tensor = torch.stack(scores)
                    index = teacher_index[str(row["qa_id"])]
                    teachers = teacher_probabilities[index].to(runtime.device)
                    reliability = None
                    if args.method == "rwkd":
                        reliability = torch.tensor(
                            [weights[(str(row["qa_id"]), expert)] for expert in experts],
                            device=runtime.device,
                        )
                    auxiliary = kd_loss(
                        score_tensor,
                        teachers,
                        temperature=temperature,
                        reliability=reliability,
                    )
                    coefficients = kd_score_gradient(
                        score_tensor,
                        teachers,
                        temperature=temperature,
                        reliability=reliability,
                    )
                    auxiliary_values.append(float(auxiliary))
                    for answer, coefficient in zip(classes, coefficients, strict=True):
                        if abs(float(coefficient.item())) < 1e-10:
                            continue
                        candidate_loss, _ = _forward(runtime, row, answer)
                        score = -candidate_loss
                        (score * coefficient / len(batch)).backward()
                    runtime.media_cache.clear()
            else:
                assert embedding_data is not None and embedding_index is not None
                assert projectors is not None
                assert loss_weight is not None
                hidden_rows = []
                for row in batch:
                    hidden_capture.clear()
                    gold_loss, metadata = _forward(runtime, row, str(row["gold_response"]))
                    (gold_loss / len(batch)).backward()
                    supervised_values.append(float(gold_loss.detach()))
                    hidden_rows.append(
                        hidden_capture["value"][
                            0, metadata["response_start"] - 1
                        ].detach()
                    )
                    runtime.media_cache.clear()
                hidden_leaf = torch.stack(hidden_rows).requires_grad_(True)
                indexes = [embedding_index[str(row["qa_id"])] for row in batch]
                expert_values = {
                    expert: torch.from_numpy(
                        embedding_data[f"{expert}_embeddings"][indexes]
                    ).to(runtime.device)
                    for expert in experts
                }
                available = {
                    expert: torch.from_numpy(
                        embedding_data[f"{expert}_available"][indexes]
                    ).to(runtime.device)
                    for expert in experts
                }
                auxiliary = unsigned_alignment_loss(
                    hidden_leaf,
                    expert_values,
                    available,
                    projectors,
                )
                (loss_weight * auxiliary).backward()
                auxiliary_values.append(float(auxiliary.detach()))
                hidden_gradients = hidden_leaf.grad.detach()
                for index, row in enumerate(batch):
                    hidden_capture.clear()
                    _, metadata = _forward(runtime, row, str(row["gold_response"]))
                    hidden = hidden_capture["value"][
                        0, metadata["response_start"] - 1
                    ]
                    hidden.backward(hidden_gradients[index])
                    runtime.media_cache.clear()
            if args.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
            optimizer.step()
            record = {
                "step": step,
                "method": args.method,
                "qa_ids": [str(row["qa_id"]) for row in batch],
                "auxiliary_loss": float(np.mean(auxiliary_values)),
                "rpu_used_by_loss": False,
            }
            if supervised_values:
                record["supervised_gold_nll"] = float(np.mean(supervised_values))
            if temperature is not None:
                record["temperature"] = temperature
            if loss_weight is not None:
                record["lambda"] = loss_weight
            log.append(record)
            write_jsonl(output / "train_log.jsonl", log)
            print(json.dumps(record), flush=True)
    finally:
        if hook is not None:
            hook.remove()
        if embedding_data is not None:
            embedding_data.close()
        runtime.close()
    adapter_dir = output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "baseline",
        "method": args.method,
        "state_dict": adapter_state(runtime.sde_modules),
        "model_config_sha256": runtime.model_config_hash,
        "protocol": runtime.protocol.name,
        "protocol_config_sha256": runtime.protocol_config_hash,
        "lora": method_config["sde"]["lora"],
        "projectors": projectors.state_dict() if projectors is not None else None,
        "seed": seed,
    }
    torch.save(payload, adapter_dir / "adapter.pt")
    summary = {
        "stage": "train_distillation",
        "method": args.method,
        "steps": steps,
        "seed": seed,
        "rows": len(rows),
        "last": log[-1],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def decode(args: argparse.Namespace) -> dict[str, Any]:
    method_config = load_method_config(args.method_config, require_resolved=False)
    runtime = QwenOmniRuntime(
        stage="sde",
        config=method_config,
        model_path=args.model_path,
        protocol_config=args.protocol_config,
        protocol_name=args.protocol_name,
    )
    adapter_file = args.adapter / "adapter.pt" if args.adapter.is_dir() else args.adapter
    payload = torch.load(adapter_file, map_location="cpu", weights_only=False)
    if payload.get("stage") != "baseline" or payload.get("method") not in METHODS:
        raise ValueError("adapter is not a supported distillation baseline")
    if payload["model_config_sha256"] != runtime.model_config_hash:
        raise ValueError("baseline adapter and model differ")
    if payload["protocol_config_sha256"] != runtime.protocol_config_hash:
        raise ValueError("baseline adapter and prompt protocol differ")
    if payload["protocol"] != runtime.protocol.name:
        raise ValueError("baseline adapter uses another answer protocol")
    if payload["lora"] != method_config["sde"]["lora"]:
        raise ValueError("baseline adapter and LoRA configuration differ")
    load_adapter_state(runtime.sde_modules, payload["state_dict"])
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    rows = read_jsonl(args.manifest)
    outputs = runtime.generate(rows)
    for row in outputs:
        row["method"] = payload["method"]
        row["normalized_prediction"] = row["answer"]
        row["terminal_status"] = "success"
    write_jsonl(args.output, outputs)
    runtime.close()
    return {
        "stage": "decode_distillation",
        "method": payload["method"],
        "rows": len(outputs),
        "output": str(args.output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--manifest", type=Path, required=True)
    prepare_command.add_argument("--labels", type=Path, required=True)
    prepare_command.add_argument("--expert-manifest", type=Path, required=True)
    prepare_command.add_argument(
        "--expert-probabilities", action="append", required=True
    )
    prepare_command.add_argument("--expert-arrays", action="append", default=[])
    prepare_command.add_argument("--sde-log", type=Path)
    prepare_command.add_argument(
        "--protocol-config",
        type=Path,
        default=ROOT / "baseline/config.yaml",
    )
    prepare_command.add_argument("--protocol-name", default="music_avqa_42")
    prepare_command.add_argument("--folds", type=int, default=5)
    prepare_command.add_argument("--seed", type=int, default=18)
    prepare_command.add_argument("--output", type=Path, required=True)
    prepare_command.set_defaults(function=prepare)

    train_command = commands.add_parser("train")
    train_command.add_argument("--method", choices=METHODS, required=True)
    train_command.add_argument("--bundle", type=Path, required=True)
    train_command.add_argument("--schedule", type=Path)
    train_command.add_argument("--model-path", type=Path, required=True)
    train_command.add_argument(
        "--method-config", type=Path, default=ROOT / "configs/qwen3b.yaml"
    )
    train_command.add_argument(
        "--baseline-config", type=Path, default=ROOT / "baseline/config.yaml"
    )
    train_command.add_argument(
        "--protocol-config", type=Path, default=ROOT / "baseline/config.yaml"
    )
    train_command.add_argument("--protocol-name", default="music_avqa_42")
    train_command.add_argument("--steps", type=int)
    train_command.add_argument("--batch-size", type=int, default=10)
    train_command.add_argument("--temperature", type=float)
    train_command.add_argument("--loss-weight", type=float)
    train_command.add_argument("--learning-rate", type=float, default=5e-5)
    train_command.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_command.add_argument("--seed", type=int)
    train_command.add_argument("--output", type=Path, required=True)
    train_command.set_defaults(function=train)

    decode_command = commands.add_parser("decode")
    decode_command.add_argument("--model-path", type=Path, required=True)
    decode_command.add_argument("--adapter", type=Path, required=True)
    decode_command.add_argument("--manifest", type=Path, required=True)
    decode_command.add_argument(
        "--method-config", type=Path, default=ROOT / "configs/qwen3b.yaml"
    )
    decode_command.add_argument(
        "--protocol-config", type=Path, default=ROOT / "baseline/config.yaml"
    )
    decode_command.add_argument("--protocol-name", default="music_avqa_42")
    decode_command.add_argument("--output", type=Path, required=True)
    decode_command.set_defaults(function=decode)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(arguments.function(arguments), indent=2))
