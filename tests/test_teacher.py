import pytest
import torch

from sde_cctdh.handoff import fit_ridge_teacher


def test_teacher_rejects_fit_use_source_overlap():
    with pytest.raises(ValueError, match="overlap"):
        fit_ridge_teacher(
            "muq",
            torch.randn(2, 3),
            torch.randn(2, 2),
            fit_source_ids=["a", "b"],
            supervision_source_ids=["b", "c"],
            ridge=1.0,
        )


def test_teacher_is_expert_specific_and_finite():
    teacher = fit_ridge_teacher(
        "muq",
        torch.randn(3, 4),
        torch.randn(3, 2),
        fit_source_ids=["a", "b", "c"],
        supervision_source_ids=["x", "y"],
        ridge=1.0,
    )
    assert teacher.expert_name == "muq"
    assert torch.isfinite(teacher.weight).all()
