#!/usr/bin/env python3
"""Train the source-disjoint AVQA four-candidate expert head."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODEL_SPECS = {
    "imagebind": {
        "media": "features/imagebind/vision.npy",
        "media_ids": "features/imagebind/clip_ids.json",
        "text": "imagebind",
        "concat": "features/imagebind/audio.npy",
    },
    "muq": {
        "media": "features/muq_videomae/muq.npy",
        "media_ids": "features/muq_videomae/clip_ids.json",
        "text": "clip",
    },
    "videomae": {
        "media": "features/muq_videomae/videomae.npy",
        "media_ids": "features/muq_videomae/clip_ids.json",
        "text": "clip",
    },
}


@dataclass(frozen=True)
class AVQAConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 6
    seed: int = 18


class AVQACandidateHead(nn.Module):
    """Shared scorer for [m, q, c, m*q, m*c, q*c]."""

    def __init__(
        self,
        media_dim: int,
        text_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.media_projection = nn.Sequential(
            nn.Linear(media_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.shared_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        media: torch.Tensor,
        question: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 3 or candidates.shape[1] != 4:
            raise ValueError("AVQA candidates must have shape [batch, 4, text]")
        m = self.media_projection(media)[:, None, :].expand(-1, 4, -1)
        q = self.text_projection(question)[:, None, :].expand(-1, 4, -1)
        c = self.text_projection(candidates)
        features = torch.cat([m, q, c, m * q, m * c, q * c], dim=-1)
        return self.shared_scorer(features).squeeze(-1)


def candidate_permutation(qa_id: str, epoch: int, seed: int) -> np.ndarray:
    hashes = [
        int(
            hashlib.sha256(f"{qa_id}|{epoch}|{seed}|{index}".encode()).hexdigest()[
                :16
            ],
            16,
        )
        for index in range(4)
    ]
    return np.argsort(hashes, kind="stable")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_protocol_data(
    root: Path,
    expert: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    spec = MODEL_SPECS[expert]
    rows = json.loads((root / "data/full_qa_manifest.json").read_text())
    qa_ids = json.loads((root / "features/text/qa_ids.json").read_text())
    question = np.load(
        root / f"features/text/{spec['text']}_question.npy",
        mmap_mode="r",
    )
    candidates = np.load(
        root / f"features/text/{spec['text']}_options.npy",
        mmap_mode="r",
    )
    media = np.load(root / spec["media"], mmap_mode="r")
    media_ids = json.loads((root / spec["media_ids"]).read_text())
    if "concat" in spec:
        second = np.load(root / spec["concat"], mmap_mode="r")
        media = np.concatenate([media, second], axis=1)
    if len(qa_ids) != len(question) or candidates.shape[:2] != (len(qa_ids), 4):
        raise ValueError("invalid AVQA text feature index")
    if len(media_ids) != len(media):
        raise ValueError("invalid AVQA media feature index")

    qa_index = {str(value): index for index, value in enumerate(qa_ids)}
    media_index = {str(value): index for index, value in enumerate(media_ids)}
    indexed_rows = []
    for row in rows:
        qa_id = str(row["qa_id"])
        clip_id = str(row["clip_id"])
        if qa_id not in qa_index or clip_id not in media_index:
            raise ValueError(f"missing frozen features for QA {qa_id}")
        indexed_rows.append(
            row
            | {
                "_question_index": qa_index[qa_id],
                "_media_index": media_index[clip_id],
            }
        )
    return indexed_rows, media, question, candidates


def _select(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def _assert_protocol_splits(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    if (len(train), len(validation)) != (4050, 450):
        raise ValueError("latest AVQA expert protocol requires 4050/450 fit/val")
    train_sources = {str(row["source_id"]) for row in train}
    validation_sources = {str(row["source_id"]) for row in validation}
    if train_sources & validation_sources:
        raise ValueError("AVQA expert fit and internal validation sources overlap")


def _assert_partition_contract(rows: list[dict[str, Any]]) -> None:
    expected = {
        "expert_train": 4050,
        "expert_internal_val": 450,
        "adapt": 3524,
        "tune": 997,
        "holdout": 997,
    }
    counts = {
        partition: sum(row["partition"] == partition for row in rows)
        for partition in expected
    }
    if counts != expected:
        raise ValueError(f"AVQA protocol partition counts differ: {counts}")
    source_sets = {
        partition: {
            str(row["source_id"])
            for row in rows
            if row["partition"] == partition
        }
        for partition in expected
    }
    partitions = list(expected)
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            if source_sets[left] & source_sets[right]:
                raise ValueError(f"AVQA source overlap between {left} and {right}")


def _fit_normalizer(
    train: list[dict[str, Any]],
    media: np.ndarray,
    question: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    media_indices = np.asarray([row["_media_index"] for row in train])
    question_indices = np.asarray([row["_question_index"] for row in train])
    media_fit = np.asarray(media[media_indices], dtype=np.float32)
    question_fit = np.asarray(question[question_indices], dtype=np.float32)
    candidate_fit = np.asarray(candidates[question_indices], dtype=np.float32)

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        return mean, std

    media_mean, media_std = moments(media_fit)
    question_mean, question_std = moments(question_fit)
    candidate_mean, candidate_std = moments(
        candidate_fit.reshape(-1, candidate_fit.shape[-1])
    )
    return {
        "media_mean": media_mean,
        "media_std": media_std,
        "question_mean": question_mean,
        "question_std": question_std,
        "candidate_mean": candidate_mean,
        "candidate_std": candidate_std,
    }


def _batch(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    media: np.ndarray,
    question: np.ndarray,
    candidates: np.ndarray,
    normalizer: dict[str, np.ndarray],
    *,
    device: torch.device,
    epoch: int | None,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    selected = [rows[int(index)] for index in indices]
    media_indices = np.asarray([row["_media_index"] for row in selected])
    question_indices = np.asarray([row["_question_index"] for row in selected])
    media_batch = (
        np.asarray(media[media_indices], dtype=np.float32)
        - normalizer["media_mean"]
    ) / normalizer["media_std"]
    question_batch = (
        np.asarray(question[question_indices], dtype=np.float32)
        - normalizer["question_mean"]
    ) / normalizer["question_std"]
    candidate_batch = (
        np.asarray(candidates[question_indices], dtype=np.float32)
        - normalizer["candidate_mean"]
    ) / normalizer["candidate_std"]
    gold = np.asarray([int(row["answer"]) for row in selected], dtype=np.int64)
    if epoch is not None:
        for index, row in enumerate(selected):
            permutation = candidate_permutation(str(row["qa_id"]), epoch, seed)
            candidate_batch[index] = candidate_batch[index, permutation]
            gold[index] = int(np.flatnonzero(permutation == gold[index])[0])
    return (
        torch.from_numpy(media_batch).to(device),
        torch.from_numpy(question_batch).to(device),
        torch.from_numpy(candidate_batch).to(device),
        torch.from_numpy(gold).to(device),
    )


@torch.no_grad()
def _infer(
    model: AVQACandidateHead,
    rows: list[dict[str, Any]],
    media: np.ndarray,
    question: np.ndarray,
    candidates: np.ndarray,
    normalizer: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    model.eval()
    records = []
    for start in range(0, len(rows), batch_size):
        indices = np.arange(start, min(start + batch_size, len(rows)))
        media_batch, question_batch, candidate_batch, gold = _batch(
            rows,
            indices,
            media,
            question,
            candidates,
            normalizer,
            device=device,
            epoch=None,
            seed=seed,
        )
        logits = model(media_batch, question_batch, candidate_batch)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for local_index, prediction in enumerate(predictions):
            row = rows[start + local_index]
            gold_index = int(row["answer"])
            records.append(
                {
                    "qa_id": str(row["qa_id"]),
                    "source_id": str(row["source_id"]),
                    "partition": str(row["partition"]),
                    "expert_answer": str(row["multi_choice"][prediction]),
                    "expert_answer_index": int(prediction),
                    "gold_answer": str(row["multi_choice"][gold_index]),
                    "gold_answer_index": gold_index,
                    "expert_correct": bool(prediction == gold_index),
                    "expert_available": True,
                    "probabilities": probabilities[local_index].tolist(),
                }
            )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_candidate_expert(
    *,
    protocol_root: Path,
    output_dir: Path,
    expert_name: str,
    config: AVQAConfig,
    device: torch.device,
    strict_protocol: bool = True,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    rows, media, question, candidates = _load_protocol_data(
        protocol_root,
        expert_name,
    )
    train_rows = _select(rows, "expert_train")
    validation_rows = _select(rows, "expert_internal_val")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"expert output directory is not empty: {output_dir}")
    if strict_protocol:
        _assert_partition_contract(rows)
        _assert_protocol_splits(train_rows, validation_rows)
    normalizer = _fit_normalizer(
        train_rows,
        media,
        question,
        candidates,
    )
    model = AVQACandidateHead(
        media.shape[1],
        question.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_checkpoint = output_dir / "best.pt"
    stale_epochs = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(
            len(train_rows)
        )
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            media_batch, question_batch, candidate_batch, gold = _batch(
                train_rows,
                batch_indices,
                media,
                question,
                candidates,
                normalizer,
                device=device,
                epoch=epoch,
                seed=config.seed,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(media_batch, question_batch, candidate_batch)
            loss = F.cross_entropy(logits, gold)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_predictions = _infer(
            model,
            validation_rows,
            media,
            question,
            candidates,
            normalizer,
            batch_size=config.batch_size,
            device=device,
            seed=config.seed,
        )
        accuracy = float(
            np.mean([row["expert_correct"] for row in validation_predictions])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "internal_validation_accuracy": accuracy,
            }
        )
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "normalizer": normalizer,
            "expert_name": expert_name,
            "epoch": epoch,
            "internal_validation_accuracy": accuracy,
            "config": config.__dict__,
        }
        torch.save(state, output_dir / "last.pt")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            stale_epochs = 0
            torch.save(state, best_checkpoint)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if not best_checkpoint.exists():
        raise RuntimeError("AVQA expert training did not produce a checkpoint")
    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    union_rows = [
        row
        for row in rows
        if row["partition"]
        in {"expert_internal_val", "adapt", "tune", "holdout"}
    ]
    predictions = _infer(
        model,
        union_rows,
        media,
        question,
        candidates,
        normalizer,
        batch_size=config.batch_size,
        device=device,
        seed=config.seed,
    )
    for row in predictions:
        row["expert_name"] = expert_name
    _write_jsonl(output_dir / "non_expert_union_predictions.jsonl", predictions)
    holdout = [row for row in predictions if row["partition"] == "holdout"]
    summary = {
        "protocol": "avqa_four_candidate_shared_scorer_v1",
        "expert": expert_name,
        "frozen_features": True,
        "train_rows": len(train_rows),
        "internal_validation_rows": len(validation_rows),
        "best_internal_validation_accuracy": best_accuracy,
        "holdout_rows": len(holdout),
        "holdout_accuracy": float(
            np.mean([row["expert_correct"] for row in holdout])
        ),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument(
        "--strict-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    config = AVQAConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    summary = train_candidate_expert(
        protocol_root=args.protocol_root,
        output_dir=args.output,
        expert_name=args.expert,
        config=config,
        device=torch.device(args.device),
        strict_protocol=args.strict_protocol,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
