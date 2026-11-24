"""Eq. 7 saliency windows, interventions, and strict validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch


Operator = Literal["mask", "replace"]


@dataclass(frozen=True)
class InterventionEffects:
    target_mask: float
    target_replace: float
    control_mask: float
    control_replace: float
    donor_mask: float
    donor_replace: float

    def as_operator_rows(self) -> tuple[tuple[float, float, float], ...]:
        return (
            (self.target_mask, self.control_mask, self.donor_mask),
            (self.target_replace, self.control_replace, self.donor_replace),
        )


def saliency_l2(gold_log_probability: torch.Tensor, local_hidden: torch.Tensor) -> torch.Tensor:
    """Return ||d log pi(y) / d H_l||_2 for every local position."""
    gradient = torch.autograd.grad(
        gold_log_probability,
        local_hidden,
        retain_graph=True,
        create_graph=False,
    )[0]
    if gradient.ndim < 2:
        raise ValueError("local hidden must have a position and feature dimension")
    return gradient.float().flatten(start_dim=1).norm(dim=1)


def top_window(saliency: torch.Tensor, window_length: int) -> torch.Tensor:
    """Select the highest-saliency contiguous window with a stable left tie."""
    if saliency.ndim != 1 or not 1 <= window_length <= saliency.numel():
        raise ValueError("invalid saliency vector or window length")
    sums = saliency.unfold(0, window_length, 1).sum(dim=1)
    start = int(torch.argmax(sums))
    return torch.arange(start, start + window_length, device=saliency.device)


def low_saliency_control_window(
    saliency: torch.Tensor,
    target_indices: torch.Tensor,
    window_length: int,
) -> torch.Tensor:
    """Select a same-source, non-overlapping low-saliency contiguous window."""
    candidates = []
    target = set(map(int, target_indices.detach().cpu().tolist()))
    for start in range(0, saliency.numel() - window_length + 1):
        indices = set(range(start, start + window_length))
        if target.isdisjoint(indices):
            candidates.append((float(saliency[start : start + window_length].sum()), start))
    if not candidates:
        raise ValueError("no non-overlapping matched control window exists")
    _, start = min(candidates)
    return torch.arange(start, start + window_length, device=saliency.device)


def intervention_operator(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    *,
    operator: Operator,
    mask_value: torch.Tensor | float = 0.0,
    replacement: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply mask or different-source replacement to one local window."""
    output = hidden.clone()
    if operator == "mask":
        output[indices] = mask_value
    elif operator == "replace":
        if replacement is None:
            raise ValueError("replace intervention requires donor content")
        if replacement.shape != output[indices].shape:
            raise ValueError("replacement and selected window shapes differ")
        output[indices] = replacement
    else:
        raise ValueError(f"unsupported operator: {operator}")
    return output


def intervention_effect(
    original_gold_log_probability: torch.Tensor,
    intervened_gold_log_probability: torch.Tensor,
) -> float:
    return float(
        original_gold_log_probability.detach()
        - intervened_gold_log_probability.detach()
    )


def validation_passes(effects: InterventionEffects) -> bool:
    """Require strict target superiority for both mask and replace."""
    for target, control, donor in effects.as_operator_rows():
        values = torch.tensor([target, control, donor], dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            return False
        if not (target > 0.0 and target > max(control, donor)):
            return False
    return True


def validate_effect_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach kappa to remote-server 3-window x 2-operator records."""
    columns = (
        "target_mask",
        "target_replace",
        "control_mask",
        "control_replace",
        "donor_mask",
        "donor_replace",
    )
    missing = set(columns) - set(rows.columns)
    if missing:
        raise ValueError(f"intervention table missing columns: {sorted(missing)}")
    out = rows.copy()
    out["kappa"] = [
        validation_passes(
            InterventionEffects(**{
                column: float(getattr(row, column))
                for column in columns
            })
        )
        for row in out.itertuples(index=False)
    ]
    return out
