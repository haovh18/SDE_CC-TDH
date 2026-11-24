#!/usr/bin/env python3
"""Runnable synthetic check of the complete model-agnostic method chain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sde_cctdh.cctdh import (
    CCTDHConfig,
    apply_cctdh_step,
    build_parent_retention_pools,
    mine_residual_repairs,
)
from sde_cctdh.handoff import fit_detail_transform, fit_ridge_teacher
from sde_cctdh.intervention import InterventionEffects, validation_passes
from sde_cctdh.metrics import evaluate_predictions
from sde_cctdh.rpu import build_rpu_manifest
from sde_cctdh.routing import (
    crossfit_group_reliability,
    route_validated_experts,
)
from sde_cctdh.sde import SDEWeights, compute_sde_objective


def main() -> None:
    pair_input = pd.DataFrame(
        [
            {
                "qa_id": "r1", "source_id": "s1", "base_answer": "no",
                "gold_answer": "yes", "expert_name": "muq",
                "expert_answer": "yes", "expert_available": True,
                "question_group": "audio",
            },
            {
                "qa_id": "p1", "source_id": "s2", "base_answer": "yes",
                "gold_answer": "yes", "expert_name": "muq",
                "expert_answer": "no", "expert_available": True,
                "question_group": "audio",
            },
            {
                "qa_id": "u1", "source_id": "s3", "base_answer": "no",
                "gold_answer": "yes", "expert_name": "muq",
                "expert_answer": "no", "expert_available": True,
                "question_group": "audio",
            },
            {
                "qa_id": "o1", "source_id": "s4", "base_answer": "yes",
                "gold_answer": "yes", "expert_name": "muq",
                "expert_answer": "yes", "expert_available": True,
                "question_group": "audio",
            },
        ]
    )
    pairs, examples = build_rpu_manifest(pair_input)
    assert examples.set_index("qa_id")["example_state"].to_dict() == {
        "r1": "REPAIR", "p1": "PROTECT", "u1": "UNCERTAIN", "o1": "NO_ACTION",
    }

    sde = compute_sde_objective(
        torch.tensor([0.4, 0.6], requires_grad=True),
        torch.tensor([0.2], requires_grad=True),
        torch.tensor([0.1], requires_grad=True),
        weights=SDEWeights(),
    )
    assert torch.isclose(sde["loss"], torch.tensor(0.8))

    parent = pd.DataFrame(
        [
            {"qa_id": "r1", "parent_answer": "no", "parent_response": "no"},
            {"qa_id": "p1", "parent_answer": "yes", "parent_response": "yes"},
            {"qa_id": "u1", "parent_answer": "no", "parent_response": "no"},
            {"qa_id": "o1", "parent_answer": "yes", "parent_response": "yes"},
        ]
    )
    residual = mine_residual_repairs(
        examples[examples["example_state"].eq("REPAIR")],
        parent,
        pairs,
    )
    protect, anchor = build_parent_retention_pools(examples, parent)
    assert residual["qa_id"].tolist() == ["r1"]
    assert protect["qa_id"].tolist() == ["p1"]
    assert anchor["qa_id"].tolist() == ["o1"]

    effects = InterventionEffects(
        target_mask=0.8, target_replace=0.7,
        control_mask=0.2, control_replace=0.1,
        donor_mask=0.1, donor_replace=0.2,
    )
    assert validation_passes(effects)

    audit_rows = []
    for index in range(10):
        audit_rows.append(
            {
                "qa_id": f"fit_{index}", "source_id": f"fit_s{index}",
                "expert_name": "muq", "question_group": "audio",
                "expert_available": True, "expert_correct": index < 8,
            }
        )
    reliability = crossfit_group_reliability(
        pd.DataFrame(audit_rows),
        n_folds=2,
        seed=18,
    )
    route_input = reliability.loc[[0], [
        "qa_id", "source_id", "question_group", "expert_name",
    ]].copy()
    route_input["expert_support"] = True
    route_input["kappa"] = True
    routed = route_validated_experts(
        route_input,
        reliability,
        tie_order=["muq"],
    )
    assert routed.loc[0, "selected_expert"] == "muq"

    generator = torch.Generator().manual_seed(18)
    raw_details = torch.randn(6, 5, generator=generator)
    transform = fit_detail_transform("muq", raw_details, latent_dim=3)
    detail_targets = transform.transform(raw_details[:4])
    hidden = torch.randn(4, 4, generator=generator)
    teacher = fit_ridge_teacher(
        "muq",
        hidden,
        detail_targets,
        fit_source_ids=["f1", "f2", "f3", "f4"],
        supervision_source_ids=["u1", "u2"],
        ridge=1.0,
    )
    assert teacher.predict(torch.randn(2, 4, generator=generator)).shape == (2, 3)

    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.0)
    basis = torch.empty(3, 0)
    result = apply_cctdh_step(
        repair_gold_loss=(parameter.square()).sum(),
        protect_parent_loss=((parameter - 0.1).square()).sum(),
        anchor_parent_loss=((parameter + 0.1).square()).sum(),
        detail_loss=-(parameter[0] * parameter[1]),
        parameters=[parameter],
        optimizer=optimizer,
        parent_retention_basis=basis,
        residual_step=8,
        config=CCTDHConfig(gradient_clip_norm=None),
    )
    assert torch.isfinite(result.final).all()

    predictions = examples[["qa_id", "source_id", "base_answer", "gold_answer", "example_state"]].copy()
    predictions["method_answer"] = predictions["base_answer"]
    predictions.loc[predictions["qa_id"].eq("r1"), "method_answer"] = "yes"
    metrics = evaluate_predictions(predictions)
    assert metrics["W2C"] == 1 and metrics["C2W"] == 0

    print(json.dumps({
        "states": examples["example_state"].value_counts().to_dict(),
        "residual_repairs": len(residual),
        "validation_pass": True,
        "teacher_latent_dim": int(teacher.weight.shape[1]),
        "detail_weight_step8": result.detail_weight,
        "metrics": metrics,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
