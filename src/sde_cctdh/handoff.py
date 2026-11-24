"""Expert detail transforms and frozen expert-specific ridge teachers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DetailTransform:
    expert_name: str
    mean: torch.Tensor
    components: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    def transform(self, detail: torch.Tensor) -> torch.Tensor:
        centered = detail.float() - self.mean.to(detail.device)
        projected = centered @ self.components.to(detail.device).T
        whitened = projected / self.std.to(detail.device)
        return F.normalize(whitened, dim=-1, eps=self.eps)


def raw_detail_direction(
    target_pooled_feature: torch.Tensor,
    control_pooled_feature: torch.Tensor,
) -> torch.Tensor:
    if target_pooled_feature.shape != control_pooled_feature.shape:
        raise ValueError("target and control pooled feature shapes differ")
    return target_pooled_feature.float() - control_pooled_feature.float()


def fit_detail_transform(
    expert_name: str,
    raw_details: torch.Tensor,
    *,
    latent_dim: int,
    eps: float = 1e-6,
) -> DetailTransform:
    if raw_details.ndim != 2 or raw_details.shape[0] < 2:
        raise ValueError("raw details must be [sample, feature] with at least two rows")
    mean = raw_details.float().mean(dim=0)
    centered = raw_details.float() - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(
        int(latent_dim),
        int((singular > max(float(singular[0]) * 1e-6, eps)).sum()),
    )
    if rank < 1:
        raise ValueError("detail directions have zero usable rank")
    std = torch.clamp(
        singular[:rank] / max(1, raw_details.shape[0] - 1) ** 0.5,
        min=eps,
    )
    return DetailTransform(
        expert_name=str(expert_name),
        mean=mean.detach(),
        components=vh[:rank].detach(),
        std=std.detach(),
        eps=eps,
    )


def build_detail_target(
    target_pooled_feature: torch.Tensor,
    control_pooled_feature: torch.Tensor,
    transform: DetailTransform,
) -> torch.Tensor:
    return transform.transform(
        raw_detail_direction(target_pooled_feature, control_pooled_feature)
    )


@dataclass(frozen=True)
class RidgeTeacher:
    expert_name: str
    weight: torch.Tensor
    ridge: float
    fit_sources: frozenset[str]

    def predict(self, hidden: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        return F.normalize(
            hidden.float() @ self.weight.to(hidden.device),
            dim=-1,
            eps=eps,
        )


def assert_source_disjoint(
    fit_source_ids: Sequence[str],
    supervision_source_ids: Sequence[str],
) -> None:
    fit = set(map(str, fit_source_ids))
    supervision = set(map(str, supervision_source_ids))
    if not fit or not supervision:
        raise ValueError("fit and supervision source sets must both be non-empty")
    overlap = fit & supervision
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"teacher fit/use source overlap: {preview}")


def fit_ridge_teacher(
    expert_name: str,
    parent_hidden: torch.Tensor,
    detail_targets: torch.Tensor,
    *,
    fit_source_ids: Sequence[str],
    supervision_source_ids: Sequence[str],
    ridge: float,
) -> RidgeTeacher:
    """Fit Eq. 8 only after proving fit/use source disjointness."""
    assert_source_disjoint(fit_source_ids, supervision_source_ids)
    if parent_hidden.ndim != 2 or detail_targets.ndim != 2:
        raise ValueError("teacher matrices must be two-dimensional")
    if parent_hidden.shape[0] != detail_targets.shape[0]:
        raise ValueError("hidden and detail row counts differ")
    if parent_hidden.shape[0] != len(fit_source_ids):
        raise ValueError("fit source IDs do not align with teacher rows")
    if ridge <= 0:
        raise ValueError("ridge coefficient must be positive")
    h = parent_hidden.float()
    z = detail_targets.float()
    identity = torch.eye(h.shape[1], device=h.device, dtype=h.dtype)
    weight = torch.linalg.solve(h.T @ h + ridge * identity, h.T @ z)
    return RidgeTeacher(
        expert_name=str(expert_name),
        weight=weight.detach(),
        ridge=float(ridge),
        fit_sources=frozenset(map(str, fit_source_ids)),
    )


def detail_cosine_loss(
    residual_hidden: torch.Tensor,
    expert_detail_target: torch.Tensor,
    teacher: RidgeTeacher,
) -> torch.Tensor:
    predicted = teacher.predict(residual_hidden)
    target = F.normalize(expert_detail_target.float(), dim=-1, eps=1e-8)
    return (1.0 - (predicted * target).sum(dim=-1)).mean()
