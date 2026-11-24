import json

import pytest
import torch

from sde_cctdh.cctdh import apply_cctdh_step
from sde_cctdh.rpu import ExampleState, PairState
from sde_cctdh.sde import apply_sde_step
from sde_cctdh.training import orthonormalize_basis, projected_optimizer_step


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ],
)
def test_rank_deficient_basis_preserves_column_space(matrix):
    source = torch.tensor(matrix)
    basis = orthonormalize_basis(source)
    assert basis.shape[1] == int(torch.linalg.matrix_rank(source))
    assert torch.allclose(basis @ (basis.T @ source), source, atol=1e-6)
    assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-6)


@pytest.mark.parametrize("stage", ["sde", "cctdh"])
def test_adamw_parameter_updates_avoid_retention_subspace(stage):
    parameters = [
        torch.nn.Parameter(torch.tensor([0.3, -0.2])),
        torch.nn.Parameter(torch.tensor([0.1])),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1)
    basis = torch.tensor([[1.0], [2.0], [3.0]]) / 14**0.5
    for step in range(1, 4):
        before = torch.cat([parameter.detach().clone() for parameter in parameters])
        value = torch.cat(parameters)
        repair = value @ torch.tensor([float(step), -2.0, 0.7])
        if stage == "sde":
            apply_sde_step(repair, parameters, optimizer, basis)
        else:
            apply_cctdh_step(
                repair_gold_loss=repair,
                protect_parent_loss=value @ torch.tensor([0.2, 0.3, -0.4]),
                anchor_parent_loss=value.sum() * 0.0,
                detail_loss=value @ torch.tensor([-0.5, 0.4, 0.2]),
                parameters=parameters,
                optimizer=optimizer,
                parent_retention_basis=basis,
                residual_step=step,
            )
        after = torch.cat([parameter.detach() for parameter in parameters])
        update = after - before
        assert update.norm() > 0
        assert (basis.T @ update).norm() / update.norm() < 1e-5
        assert int(optimizer.state[parameters[0]]["step"]) == step


def test_empty_basis_keeps_adamw_updates_unchanged():
    actual = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
    expected = torch.nn.Parameter(actual.detach().clone())
    optimizer = torch.optim.AdamW([actual], lr=0.01, weight_decay=0.1)
    reference = torch.optim.AdamW([expected], lr=0.01, weight_decay=0.1)
    for step in range(1, 4):
        gradient = torch.tensor([float(step), -0.5])
        actual.grad = gradient.clone()
        expected.grad = gradient.clone()
        projected_optimizer_step([actual], optimizer, torch.empty(2, 0))
        reference.step()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("state", list(PairState) + list(ExampleState))
def test_rpu_enums_keep_string_behavior(state):
    assert str(state) == state.value
    assert f"{state}" == state.value
    assert state == state.value
    assert json.loads(json.dumps(state)) == state.value
