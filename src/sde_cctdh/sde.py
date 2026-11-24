"""Eq. 4 and Eq. 5 for Signed Decision Editing."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd
import torch

from .rpu import ExampleState
from .training import (
    assign_flat_gradient,
    flat_autograd,
    project_away,
    projected_optimizer_step,
)


@dataclass(frozen=True)
class SDEWeights:
    repair: float = 1.0
    protect: float = 1.0
    anchor: float = 1.0

    def __post_init__(self) -> None:
        if min(self.repair, self.protect, self.anchor) < 0:
            raise ValueError("SDE weights must be non-negative")


def _pool_mean(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional loss tensor")
    return values.mean()


def compute_sde_objective(
    repair_gold_nll: torch.Tensor,
    protect_base_response_nll: torch.Tensor,
    anchor_base_response_nll: torch.Tensor,
    *,
    weights: SDEWeights = SDEWeights(),
) -> dict[str, torch.Tensor]:
    """Use a separate mean for each fixed pool; Uncertain has no argument."""
    repair = _pool_mean(repair_gold_nll, "repair_gold_nll")
    protect = _pool_mean(protect_base_response_nll, "protect_base_response_nll")
    anchor = _pool_mean(anchor_base_response_nll, "anchor_base_response_nll")
    total = weights.repair * repair + weights.protect * protect + weights.anchor * anchor
    return {
        "loss": total,
        "repair_loss": repair,
        "protect_loss": protect,
        "anchor_loss": anchor,
    }


def build_sde_pools(
    example_manifest: pd.DataFrame,
    response_cache: pd.DataFrame,
    *,
    repair_ids: Sequence[str] | None = None,
    protect_ids: Sequence[str] | None = None,
    anchor_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create fixed Eq. 4 pools while preserving complete response targets."""
    required_examples = {
        "qa_id",
        "source_id",
        "example_state",
        "gold_answer",
    }
    required_cache = {"qa_id", "gold_response", "base_response"}
    if required_examples - set(example_manifest.columns):
        raise ValueError("example manifest lacks SDE pool fields")
    if required_cache - set(response_cache.columns):
        raise ValueError("response cache lacks complete gold/base responses")
    if example_manifest["qa_id"].duplicated().any() or response_cache["qa_id"].duplicated().any():
        raise ValueError("SDE pool inputs must be unique by qa_id")
    cache_columns = [
        column
        for column in response_cache.columns
        if column == "qa_id" or column not in example_manifest.columns
    ]
    merged = example_manifest.merge(
        response_cache[cache_columns],
        on="qa_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["gold_response", "base_response"]].isna().any().any():
        raise ValueError("response cache does not cover the example manifest")

    def select(
        state: ExampleState,
        ids: Sequence[str] | None,
        target_column: str,
    ) -> pd.DataFrame:
        pool = merged[merged["example_state"].eq(state.value)].copy()
        if ids is not None:
            requested = list(map(str, ids))
            if len(requested) != len(set(requested)):
                raise ValueError(f"duplicate IDs requested for {state.value}")
            by_id = pool.set_index(pool["qa_id"].astype(str), drop=False)
            missing = set(requested) - set(by_id.index)
            if missing:
                raise ValueError(
                    f"{state.value} pool contains IDs outside its state: {sorted(missing)[:5]}"
                )
            pool = by_id.loc[requested].reset_index(drop=True)
        pool["target_response"] = pool[target_column]
        return pool.reset_index(drop=True)

    repair = select(ExampleState.REPAIR, repair_ids, "gold_response")
    protect = select(ExampleState.PROTECT, protect_ids, "base_response")
    anchor = select(ExampleState.NO_ACTION, anchor_ids, "base_response")
    return repair, protect, anchor


def apply_sde_step(
    objective: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    retention_basis: torch.Tensor,
    *,
    clip_norm: float | None = None,
) -> dict[str, float]:
    """Project the SDE gradient before the optimizer consumes it."""
    optimizer.zero_grad(set_to_none=True)
    raw = flat_autograd(objective, parameters)
    projected = project_away(raw, retention_basis)
    if not bool(torch.isfinite(projected).all()):
        raise FloatingPointError("non-finite projected SDE gradient")
    scale = 1.0
    if clip_norm is not None:
        scale = min(1.0, float(clip_norm) / (float(projected.norm()) + 1e-12))
    assign_flat_gradient(parameters, projected * scale)
    projected_optimizer_step(parameters, optimizer, retention_basis)
    return {
        "raw_gradient_norm": float(raw.norm()),
        "projected_gradient_norm": float(projected.norm()),
        "gradient_clip_scale": float(scale),
    }
