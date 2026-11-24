"""Eq. 9-Eq. 11 CC-TDH loss and gradient execution."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd
import torch

from .rpu import ExampleState, PairState
from .training import (
    assign_flat_gradient,
    flat_autograd,
    project_away,
    projected_optimizer_step,
)


def mine_residual_repairs(
    r_used: pd.DataFrame,
    parent_outputs: pd.DataFrame,
    pair_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Implement Eq. 6 using only the fixed first-stage Repair exposure."""
    required_used = {"qa_id", "source_id", "gold_answer", "example_state"}
    required_parent = {"qa_id", "parent_answer", "parent_response"}
    required_pairs = {
        "qa_id",
        "expert_name",
        "expert_available",
        "expert_correct",
        "pair_state",
    }
    for name, frame, required in (
        ("r_used", r_used, required_used),
        ("parent_outputs", parent_outputs, required_parent),
        ("pair_audit", pair_audit, required_pairs),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if not r_used["example_state"].eq(ExampleState.REPAIR.value).all():
        raise ValueError("r_used contains a non-Repair example")
    if r_used["qa_id"].duplicated().any() or parent_outputs["qa_id"].duplicated().any():
        raise ValueError("r_used and parent outputs must be unique by qa_id")

    joined = r_used.merge(
        parent_outputs[list(required_parent)],
        on="qa_id",
        how="left",
        validate="one_to_one",
    )
    if joined["parent_answer"].isna().any():
        raise ValueError("parent outputs do not cover all used Repair examples")

    support = pair_audit[
        pair_audit["expert_available"].astype(bool)
        & pair_audit["expert_correct"].astype(bool)
        & pair_audit["pair_state"].eq(PairState.REPAIR.value)
    ]
    experts = (
        support.groupby("qa_id")["expert_name"]
        .agg(lambda values: sorted(set(map(str, values))))
        .rename("supporting_experts")
    )
    joined = joined.join(experts, on="qa_id")
    joined["supporting_experts"] = joined["supporting_experts"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    return joined[
        joined["parent_answer"].ne(joined["gold_answer"])
        & joined["supporting_experts"].map(bool)
    ].reset_index(drop=True)


def build_parent_retention_pools(
    example_manifest: pd.DataFrame,
    parent_outputs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build parent-correct P_S and A_S with raw parent-response targets."""
    required_examples = {"qa_id", "source_id", "gold_answer", "example_state"}
    required_parent = {"qa_id", "parent_answer", "parent_response"}
    if required_examples - set(example_manifest.columns):
        raise ValueError("example manifest lacks retention fields")
    if required_parent - set(parent_outputs.columns):
        raise ValueError("parent outputs lack retention fields")
    joined = example_manifest.merge(
        parent_outputs[list(required_parent)],
        on="qa_id",
        how="left",
        validate="one_to_one",
    )
    parent_correct = joined["parent_answer"].eq(joined["gold_answer"])
    protect = joined[
        joined["example_state"].eq(ExampleState.PROTECT.value) & parent_correct
    ].copy()
    anchor = joined[
        joined["example_state"].eq(ExampleState.NO_ACTION.value) & parent_correct
    ].copy()
    protect["retention_target_response"] = protect["parent_response"]
    anchor["retention_target_response"] = anchor["parent_response"]
    return protect.reset_index(drop=True), anchor.reset_index(drop=True)


def build_validated_repair_pool(
    residual_repairs: pd.DataFrame,
    routed_experts: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only residual Repair examples with one validated expert route."""
    required_route = {"qa_id", "selected_expert", "kappa"}
    if required_route - set(routed_experts.columns):
        raise ValueError("routed expert table is incomplete")
    passing = routed_experts[routed_experts["kappa"].astype(bool)]
    if passing["qa_id"].duplicated().any():
        raise ValueError("router must emit at most one expert per QA")
    merge_keys = ["qa_id"]
    if "source_id" in residual_repairs and "source_id" in passing:
        merge_keys.append("source_id")
        route_source = passing.set_index("qa_id")["source_id"].astype(str)
        residual_source = residual_repairs.set_index("qa_id")["source_id"].astype(str)
        common = route_source.index.intersection(residual_source.index)
        if route_source.loc[common].ne(residual_source.loc[common]).any():
            raise ValueError("residual and routed tables disagree on source_id")
    route_columns = merge_keys + [
        column
        for column in passing.columns
        if column not in residual_repairs.columns and column not in merge_keys
    ]
    return residual_repairs.merge(
        passing[route_columns],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )


@dataclass(frozen=True)
class CCTDHConfig:
    protect_weight: float = 1.0
    anchor_weight: float = 1.0
    detail_max_weight: float = 0.015
    detail_warmup_steps: int = 8
    norm_cap_ratio: float = 0.20
    epsilon: float = 1e-12
    gradient_clip_norm: float | None = None

    def detail_weight(self, residual_step: int) -> float:
        if residual_step < 1:
            raise ValueError("residual step starts at one")
        if self.detail_warmup_steps <= 1:
            return self.detail_max_weight
        progress = min(1.0, (residual_step - 1) / (self.detail_warmup_steps - 1))
        return self.detail_max_weight * progress


@dataclass(frozen=True)
class CCTDHGradients:
    main: torch.Tensor
    retention: torch.Tensor
    detail_raw: torch.Tensor
    detail_calibrated: torch.Tensor
    detail_capped: torch.Tensor
    final: torch.Tensor
    detail_weight: float
    conflict_triggered: bool
    norm_cap_triggered: bool
    fallback: bool
    projection_removed_ratio: float
    gradient_clip_scale: float


def _conflict_calibrate(
    detail: torch.Tensor,
    retention: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, bool]:
    dot = torch.dot(detail, retention)
    triggered = bool(dot < 0)
    if not triggered:
        return detail, False
    return (
        detail
        - dot / (torch.dot(retention, retention) + epsilon) * retention,
        True,
    )


def _relative_norm_cap(
    detail: torch.Tensor,
    main: torch.Tensor,
    ratio: float,
    epsilon: float,
) -> tuple[torch.Tensor, bool]:
    cap = ratio * main.norm()
    triggered = bool(detail.norm() > cap)
    if not triggered:
        return detail, False
    return detail * (cap / (detail.norm() + epsilon)), True


def combine_cctdh_gradients(
    *,
    repair_grad: torch.Tensor,
    protect_grad: torch.Tensor,
    anchor_grad: torch.Tensor,
    detail_grad: torch.Tensor,
    parent_retention_basis: torch.Tensor,
    residual_step: int,
    config: CCTDHConfig = CCTDHConfig(),
) -> CCTDHGradients:
    """Apply conflict removal, raw-detail cap, lambda, then total projection."""
    retention = (
        config.protect_weight * protect_grad
        + config.anchor_weight * anchor_grad
    )
    main = repair_grad + retention
    calibrated, conflict = _conflict_calibrate(
        detail_grad, retention, config.epsilon
    )
    capped, cap_triggered = _relative_norm_cap(
        calibrated, main, config.norm_cap_ratio, config.epsilon
    )
    detail_weight = config.detail_weight(residual_step)
    fallback = not bool(
        torch.isfinite(detail_grad).all()
        and torch.isfinite(capped).all()
        and capped.norm() > config.epsilon
    )
    combined = main if fallback else main + detail_weight * capped
    projected = project_away(combined, parent_retention_basis)
    if not bool(torch.isfinite(projected).all()):
        fallback = True
        projected = project_away(main, parent_retention_basis)
    removed_ratio = float(
        (combined - projected).norm() / combined.norm().clamp_min(config.epsilon)
    )
    clip_scale = 1.0
    if config.gradient_clip_norm is not None:
        clip_scale = min(
            1.0,
            config.gradient_clip_norm
            / (float(projected.norm()) + config.epsilon),
        )
    final = projected * clip_scale
    return CCTDHGradients(
        main=main,
        retention=retention,
        detail_raw=detail_grad,
        detail_calibrated=calibrated,
        detail_capped=capped,
        final=final,
        detail_weight=float(detail_weight),
        conflict_triggered=conflict,
        norm_cap_triggered=cap_triggered,
        fallback=fallback,
        projection_removed_ratio=removed_ratio,
        gradient_clip_scale=float(clip_scale),
    )


def apply_cctdh_step(
    *,
    repair_gold_loss: torch.Tensor,
    protect_parent_loss: torch.Tensor,
    anchor_parent_loss: torch.Tensor,
    detail_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    parent_retention_basis: torch.Tensor,
    residual_step: int,
    config: CCTDHConfig = CCTDHConfig(),
) -> CCTDHGradients:
    """Compute independent Eq. 10 gradients and update residual LoRA only."""
    optimizer.zero_grad(set_to_none=True)
    repair_grad = flat_autograd(repair_gold_loss, parameters, retain_graph=True)
    protect_grad = flat_autograd(protect_parent_loss, parameters, retain_graph=True)
    anchor_grad = flat_autograd(anchor_parent_loss, parameters, retain_graph=True)
    detail_grad = flat_autograd(detail_loss, parameters)
    result = combine_cctdh_gradients(
        repair_grad=repair_grad,
        protect_grad=protect_grad,
        anchor_grad=anchor_grad,
        detail_grad=detail_grad,
        parent_retention_basis=parent_retention_basis,
        residual_step=residual_step,
        config=config,
    )
    assign_flat_gradient(parameters, result.final)
    projected_optimizer_step(parameters, optimizer, parent_retention_basis)
    return result
