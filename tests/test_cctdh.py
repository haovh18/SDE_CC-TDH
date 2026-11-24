import torch

from sde_cctdh.cctdh import CCTDHConfig, combine_cctdh_gradients


def test_lambda_is_applied_after_raw_detail_cap():
    config = CCTDHConfig(
        detail_max_weight=0.1,
        detail_warmup_steps=1,
        norm_cap_ratio=0.2,
        gradient_clip_norm=None,
    )
    result = combine_cctdh_gradients(
        repair_grad=torch.tensor([1.0, 0.0]),
        protect_grad=torch.zeros(2),
        anchor_grad=torch.zeros(2),
        detail_grad=torch.tensor([0.0, 10.0]),
        parent_retention_basis=torch.empty(2, 0),
        residual_step=1,
        config=config,
    )
    assert torch.allclose(result.detail_capped, torch.tensor([0.0, 0.2]))
    assert torch.allclose(result.final, torch.tensor([1.0, 0.02]))


def test_conflicting_detail_is_orthogonalized_to_retention():
    result = combine_cctdh_gradients(
        repair_grad=torch.tensor([0.0, 1.0]),
        protect_grad=torch.tensor([1.0, 0.0]),
        anchor_grad=torch.zeros(2),
        detail_grad=torch.tensor([-1.0, 1.0]),
        parent_retention_basis=torch.empty(2, 0),
        residual_step=8,
        config=CCTDHConfig(gradient_clip_norm=None),
    )
    assert result.conflict_triggered
    assert abs(float(torch.dot(result.detail_calibrated, result.retention))) < 1e-6

