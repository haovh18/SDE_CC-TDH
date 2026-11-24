"""Shared response-loss, flat-gradient, and retention-subspace utilities."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def response_nlls(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute Eq. 1 loss for each item after the standard causal shift."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [batch, seq, vocab] and labels [batch, seq]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit and label sequence shapes differ")

    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=ignore_index,
    )
    valid = shifted_labels.ne(ignore_index)
    lengths = valid.sum(dim=1)
    if bool(lengths.eq(0).any()):
        raise ValueError("every response must contain at least one supervised token")
    return (token_loss * valid).sum(dim=1) / lengths


def mean_response_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    return response_nlls(logits, labels, ignore_index=ignore_index).mean()


def flat_autograd(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = [
        grad.detach().float().reshape(-1)
        if grad is not None
        else torch.zeros(parameter.numel(), device=parameter.device)
        for parameter, grad in zip(parameters, grads, strict=True)
    ]
    return torch.cat(pieces)


def assign_flat_gradient(
    parameters: Sequence[torch.nn.Parameter],
    gradient: torch.Tensor,
) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = gradient[offset : offset + count].reshape_as(parameter).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        offset += count
    if offset != gradient.numel():
        raise ValueError("flat gradient size does not match parameters")


def orthonormalize_basis(basis: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError("basis must be [parameter, rank]")
    if basis.numel() == 0:
        return basis
    u, singular, _ = torch.linalg.svd(basis.float(), full_matrices=False)
    threshold = max(float(singular[0]) * 1e-6, eps)
    return u[:, singular.gt(threshold)].contiguous()


def signed_sketch_basis(
    gradients: torch.Tensor,
    *,
    rank: int,
    seed: int,
) -> torch.Tensor:
    """Build a seeded randomized low-rank basis from row-wise gradients."""
    if gradients.ndim != 2 or gradients.shape[0] == 0:
        raise ValueError("gradients must be a non-empty [sample, parameter] matrix")
    rank = min(int(rank), gradients.shape[0], gradients.shape[1])
    if rank < 1:
        raise ValueError("rank must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (rank, gradients.shape[0]),
        generator=generator,
        dtype=torch.int8,
    ).float()
    signs = signs.mul_(2).sub_(1).to(gradients.device)
    sketch = signs @ gradients.float() / gradients.shape[0] ** 0.5
    return orthonormalize_basis(sketch.T)


def project_away(gradient: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Compute (I - B B^T) g for an orthonormal column basis B."""
    if basis.numel() == 0:
        return gradient
    if basis.ndim != 2 or basis.shape[0] != gradient.numel():
        raise ValueError("basis must have shape [gradient.numel(), rank]")
    return gradient - basis @ (basis.T @ gradient)


@torch.no_grad()
def projected_optimizer_step(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    retention_basis: torch.Tensor,
) -> None:
    if retention_basis.numel() == 0:
        optimizer.step()
        return
    before = torch.cat([parameter.detach().float().reshape(-1) for parameter in parameters])
    optimizer.step()
    after = torch.cat([parameter.detach().float().reshape(-1) for parameter in parameters])
    updated = before + project_away(after - before, retention_basis)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.copy_(updated[offset : offset + count].reshape_as(parameter))
        offset += count
