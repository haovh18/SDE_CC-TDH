#!/usr/bin/env python3
"""Train source-grouped OOF MUSIC-AVQA expert heads over 42 answers."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


FIXED_EPOCHS = {"imagebind": 9, "muq": 5, "videomae": 14}


@dataclass(frozen=True)
class MusicHeadConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    folds: int = 5
    seed: int = 18


class MusicAVQA42Head(nn.Module):
    """Paper MUSIC head: [m, q, m*q, abs(m-q)] -> 42 logits."""

    def __init__(
        self,
        media_dim: int,
        question_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.media_projection = nn.Sequential(
            nn.Linear(media_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.question_projection = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 42),
        )

    def forward(
        self,
        media: torch.Tensor,
        question: torch.Tensor,
    ) -> torch.Tensor:
        m = self.media_projection(media)
        q = self.question_projection(question)
        features = torch.cat([m, q, m * q, torch.abs(m - q)], dim=-1)
        return self.classifier(features)


class _FrozenFeatureDataset(Dataset):
    def __init__(
        self,
        media: np.ndarray,
        question: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        normalizer: dict[str, np.ndarray],
    ) -> None:
        self.media = media
        self.question = question
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        media = (
            np.asarray(self.media[index], dtype=np.float32)
            - self.normalizer["media_mean"]
        ) / self.normalizer["media_std"]
        question = (
            np.asarray(self.question[index], dtype=np.float32)
            - self.normalizer["question_mean"]
        ) / self.normalizer["question_std"]
        return (
            torch.from_numpy(media),
            torch.from_numpy(question),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _fit_normalizer(
    media: np.ndarray,
    question: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    media_fit = np.asarray(media[fit_indices], dtype=np.float32)
    question_fit = np.asarray(question[fit_indices], dtype=np.float32)
    media_std = media_fit.std(axis=0)
    question_std = question_fit.std(axis=0)
    media_std[media_std < 1e-6] = 1.0
    question_std[question_std < 1e-6] = 1.0
    return {
        "media_mean": media_fit.mean(axis=0),
        "media_std": media_std,
        "question_mean": question_fit.mean(axis=0),
        "question_std": question_std,
    }


@torch.no_grad()
def _predict(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for media, question, target, row_index in loader:
        output = model(media.to(device), question.to(device))
        logits.append(output.float().cpu().numpy())
        labels.append(target.numpy())
        indices.append(row_index.numpy())
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(indices)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("manifest must be parquet, jsonl, or csv")


def _load_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as values:
        required = {
            "media",
            "question",
            "y",
            "train_idx",
            "val_idx",
            "test_idx",
            "vocab",
        }
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"expert array file missing fields: {sorted(missing)}")
        result = {key: values[key] for key in required}
    if len(result["vocab"]) != 42:
        raise ValueError("MUSIC-AVQA expert vocabulary must contain 42 answers")
    if result["media"].shape[0] != result["question"].shape[0]:
        raise ValueError("media and question row counts differ")
    if len(result["y"]) != result["media"].shape[0]:
        raise ValueError("label and feature row counts differ")
    return result


def train_oof(
    *,
    arrays_path: Path,
    manifest_path: Path,
    output_dir: Path,
    expert_name: str,
    config: MusicHeadConfig,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as error:
        raise RuntimeError(
            "MUSIC OOF training requires the optional 'expert' dependencies"
        ) from error

    arrays = _load_arrays(arrays_path)
    if int(arrays["y"].min()) < 0 or int(arrays["y"].max()) >= 42:
        raise ValueError("MUSIC-AVQA labels must be in the range 0..41")
    manifest = _read_manifest(manifest_path)
    required_manifest = {"qa_id", "source_id"}
    if required_manifest - set(manifest.columns):
        raise ValueError("manifest must contain qa_id and source_id")
    if len(manifest) != len(arrays["y"]):
        raise ValueError("manifest must align row-for-row with the NPZ arrays")

    train_indices = arrays["train_idx"].astype(np.int64)
    labels = arrays["y"].astype(np.int64)
    train_labels = labels[train_indices]
    train_sources = manifest.iloc[train_indices]["source_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.seed,
    )
    folds = list(
        splitter.split(
            np.zeros(len(train_indices)),
            train_labels,
            train_sources,
        )
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"expert output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_probabilities = np.zeros((len(train_indices), 42), dtype=np.float32)
    validation_probabilities = np.zeros((len(arrays["val_idx"]), 42), dtype=np.float32)
    test_probabilities = np.zeros((len(arrays["test_idx"]), 42), dtype=np.float32)
    fold_metrics = []

    for fold, (fit_local, held_local) in enumerate(folds):
        _seed_everything(config.seed + fold)
        fit_indices = train_indices[fit_local]
        held_indices = train_indices[held_local]
        normalizer = _fit_normalizer(
            arrays["media"],
            arrays["question"],
            fit_indices,
        )
        model = MusicAVQA42Head(
            arrays["media"].shape[1],
            arrays["question"].shape[1],
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        fit_dataset = _FrozenFeatureDataset(
            arrays["media"],
            arrays["question"],
            labels,
            fit_indices,
            normalizer,
        )
        for epoch in range(1, epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(
                config.seed + fold * 100 + epoch
            )
            loader = DataLoader(
                fit_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
            )
            for media, question, target, _ in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(media.to(device), question.to(device))
                loss = F.cross_entropy(logits, target.to(device))
                loss.backward()
                optimizer.step()

        split_specs = {
            "heldout": held_indices,
            "validation": arrays["val_idx"].astype(np.int64),
            "test": arrays["test_idx"].astype(np.int64),
        }
        fold_predictions = {}
        for split, indices in split_specs.items():
            dataset = _FrozenFeatureDataset(
                arrays["media"],
                arrays["question"],
                labels,
                indices,
                normalizer,
            )
            logits, targets, row_indices = _predict(
                model,
                dataset,
                batch_size=config.batch_size,
                device=device,
            )
            probabilities = _softmax(logits)
            fold_predictions[split] = probabilities
            np.savez_compressed(
                output_dir / f"{expert_name}_fold{fold}_{split}.npz",
                logits=logits.astype(np.float32),
                probabilities=probabilities,
                labels=targets,
                row_indices=row_indices,
                vocab=arrays["vocab"],
            )

        oof_probabilities[held_local] = fold_predictions["heldout"]
        validation_probabilities += fold_predictions["validation"] / config.folds
        test_probabilities += fold_predictions["test"] / config.folds
        group_overlap = set(train_sources[fit_local]) & set(train_sources[held_local])
        if group_overlap:
            raise RuntimeError("source leakage detected inside an OOF fold")
        held_accuracy = float(
            (
                fold_predictions["heldout"].argmax(axis=1)
                == labels[held_indices]
            ).mean()
        )
        fold_metrics.append(
            {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "heldout_rows": len(held_indices),
                "heldout_accuracy": held_accuracy,
            }
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "normalizer": normalizer,
                "expert_name": expert_name,
                "fold": fold,
                "epochs": epochs,
                "config": asdict(config),
                "vocab": arrays["vocab"].tolist(),
            },
            output_dir / f"{expert_name}_fold{fold}.pt",
        )

    merged = {
        "train_oof": (oof_probabilities, train_indices),
        "validation": (validation_probabilities, arrays["val_idx"]),
        "test": (test_probabilities, arrays["test_idx"]),
    }
    for split, (probabilities, row_indices) in merged.items():
        np.savez_compressed(
            output_dir / f"{expert_name}_{split}.npz",
            probabilities=probabilities,
            row_indices=row_indices,
            vocab=arrays["vocab"],
        )
        selected = manifest.iloc[np.asarray(row_indices, dtype=np.int64)].copy()
        predictions = probabilities.argmax(axis=1)
        prediction_rows = selected[["qa_id", "source_id"]].reset_index(drop=True)
        if "question_group" in selected:
            prediction_rows["question_group"] = selected[
                "question_group"
            ].astype(str).to_numpy()
        prediction_rows["expert_name"] = expert_name
        prediction_rows["expert_answer"] = [
            str(arrays["vocab"][index]) for index in predictions
        ]
        prediction_rows["gold_answer"] = [
            str(arrays["vocab"][index])
            for index in labels[np.asarray(row_indices, dtype=np.int64)]
        ]
        prediction_rows["expert_available"] = True
        prediction_rows["expert_correct"] = (
            predictions == labels[np.asarray(row_indices, dtype=np.int64)]
        )
        prediction_rows["expert_confidence"] = probabilities.max(axis=1)
        prediction_rows.to_json(
            output_dir / f"{expert_name}_{split}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
    summary = {
        "protocol": "music_avqa_42_source_grouped_oof",
        "expert": expert_name,
        "classes": 42,
        "folds": config.folds,
        "epochs": epochs,
        "fold_metrics": fold_metrics,
        "oof_accuracy": float(
            (oof_probabilities.argmax(axis=1) == train_labels).mean()
        ),
    }
    (output_dir / f"{expert_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expert",
        choices=sorted(FIXED_EPOCHS),
        required=True,
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    config = MusicHeadConfig(seed=args.seed, folds=args.folds)
    summary = train_oof(
        arrays_path=args.arrays,
        manifest_path=args.manifest,
        output_dir=args.output,
        expert_name=args.expert,
        config=config,
        epochs=args.epochs or FIXED_EPOCHS[args.expert],
        device=torch.device(args.device),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
