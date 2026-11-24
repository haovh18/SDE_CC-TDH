"""Runtime loading and deterministic batching for model-specific training adapters."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from .training import signed_sketch_basis


class TrainingRuntime(Protocol):
    optimizer: torch.optim.Optimizer

    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...

    def response_nll(
        self,
        rows: list[dict[str, Any]],
        target_field: str,
    ) -> torch.Tensor: ...

    def retention_gradients(
        self,
        rows: list[dict[str, Any]],
        target_field: str,
        batch_size: int,
    ) -> torch.Tensor: ...

    def save(self, output_dir: Path) -> None: ...


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {path}")


def parse_runtime_args(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError("--runtime-arg must be KEY=JSON_VALUE")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def load_runtime(
    spec: str,
    *,
    stage: str,
    config: dict[str, Any],
    runtime_args: dict[str, Any],
) -> TrainingRuntime:
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("--runtime must use MODULE:FACTORY syntax")
    factory = getattr(importlib.import_module(module_name), factory_name)
    runtime = factory(stage=stage, config=config, **runtime_args)
    required = (
        "optimizer",
        "trainable_parameters",
        "response_nll",
        "retention_gradients",
        "save",
    )
    missing = [name for name in required if not hasattr(runtime, name)]
    if missing:
        raise TypeError(f"training runtime lacks required members: {missing}")
    parameters = list(runtime.trainable_parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("runtime must expose non-empty trainable LoRA parameters")
    optimizer_parameters = {
        id(parameter)
        for group in runtime.optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != {id(parameter) for parameter in parameters}:
        raise ValueError("optimizer must contain exactly the exposed LoRA parameters")
    return runtime


@dataclass
class BatchCycle:
    frame: pd.DataFrame
    batch_size: int
    seed: int
    _epoch: int = 0
    _offset: int = 0
    _order: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frame.empty:
            raise ValueError("training pool must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")

    def next(self) -> list[dict[str, Any]]:
        if self._order is None or self._offset >= len(self._order):
            generator = np.random.default_rng(self.seed + self._epoch)
            self._order = generator.permutation(len(self.frame))
            self._offset = 0
            self._epoch += 1
        indices = self._order[self._offset : self._offset + self.batch_size]
        self._offset += len(indices)
        return self.frame.iloc[indices].to_dict(orient="records")


def load_or_build_basis(
    *,
    path: Path | None,
    runtime: TrainingRuntime,
    rows: pd.DataFrame,
    target_field: str,
    batch_size: int,
    rank: int,
    seed: int,
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    device = parameters[0].device
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if path is not None:
        value = torch.load(path, map_location=device, weights_only=False)
        basis = value["basis"] if isinstance(value, dict) and "basis" in value else value
        basis = torch.as_tensor(basis, device=device, dtype=torch.float32)
    else:
        gradients = runtime.retention_gradients(
            rows.to_dict(orient="records"),
            target_field,
            batch_size,
        )
        gradients = torch.as_tensor(gradients, device=device, dtype=torch.float32)
        basis = signed_sketch_basis(gradients, rank=rank, seed=seed)
    if basis.ndim != 2 or basis.shape[0] != parameter_count:
        raise ValueError(
            f"retention basis must be [{parameter_count}, rank], got {tuple(basis.shape)}"
        )
    return basis


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
