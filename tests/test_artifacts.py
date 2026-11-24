import pandas as pd
import pytest

from sde_cctdh.artifacts import (
    build_expert_pair_input,
    build_response_cache,
    fit_teacher_bundle,
    prepare_cctdh_artifacts,
)
from sde_cctdh.cctdh import build_validated_repair_pool


def test_response_cache_preserves_runtime_metadata():
    manifest = pd.DataFrame(
        [{
            "qa_id": "q1",
            "source_id": "s1",
            "gold_answer": "violin",
                "media_path": "data/one.mp4",
            "question": "What is played?",
        }]
    )
    outputs = pd.DataFrame(
        [{"qa_id": "q1", "base_answer": "violin", "base_response": "violin"}]
    )
    cache = build_response_cache(manifest, outputs)
    assert cache.loc[0, "gold_response"] == "violin"
    assert cache.loc[0, "media_path"] == "data/one.mp4"


def test_expert_pairs_keep_question_group_for_oof_routing():
    base = pd.DataFrame(
        [{
            "qa_id": "q1",
            "source_id": "s1",
            "gold_answer": "yes",
            "base_answer": "no",
        }]
    )
    experts = pd.DataFrame(
        [{
            "qa_id": "q1",
            "source_id": "s1",
            "gold_answer": "yes",
            "question_group": "audio",
            "expert_name": "muq",
            "expert_answer": "yes",
            "expert_available": True,
        }]
    )
    metadata = pd.DataFrame([{"qa_id": "q1", "question_group": "audio"}])
    pairs = build_expert_pair_input(base, experts, metadata)
    assert pairs.loc[0, "question_group"] == "audio"


def test_teacher_bundle_rejects_fit_use_source_overlap():
    fit = pd.DataFrame(
        [{
            "source_id": "s1",
            "expert_name": "muq",
            "parent_hidden": [1.0, 0.0],
            "detail_target": [0.0, 1.0],
        }]
    )
    supervision = pd.DataFrame(
        [{"source_id": "s1", "selected_expert": "muq"}]
    )
    with pytest.raises(ValueError, match="overlap"):
        fit_teacher_bundle(
            fit,
            supervision,
            hidden_layer=30,
            ridge=1.0,
        )


def test_validated_repair_keeps_single_source_column():
    residual = pd.DataFrame(
        [{"qa_id": "q1", "source_id": "s1", "gold_answer": "yes"}]
    )
    routed = pd.DataFrame(
        [{
            "qa_id": "q1",
            "source_id": "s1",
            "selected_expert": "muq",
            "question_group": "audio",
            "kappa": True,
        }]
    )
    result = build_validated_repair_pool(residual, routed)
    assert result.loc[0, "source_id"] == "s1"
    assert "source_id_x" not in result
    assert "source_id_y" not in result


def test_validated_repair_rejects_source_mismatch():
    residual = pd.DataFrame(
        [{"qa_id": "q1", "source_id": "source-a", "gold_answer": "yes"}]
    )
    routed = pd.DataFrame(
        [{
            "qa_id": "q1",
            "source_id": "source-b",
            "selected_expert": "muq",
            "kappa": True,
        }]
    )
    with pytest.raises(ValueError, match="disagree on source_id"):
        build_validated_repair_pool(residual, routed)


def test_prepare_cctdh_artifacts_connects_parent_route_and_detail():
    repair = pd.DataFrame(
        [{
            "qa_id": "q0",
            "source_id": "s0",
            "gold_answer": "yes",
            "gold_response": "yes",
            "example_state": "REPAIR",
        }]
    )
    examples = pd.DataFrame(
        [
            repair.iloc[0].to_dict(),
            {
                "qa_id": "p",
                "source_id": "sp",
                "gold_answer": "yes",
                "example_state": "PROTECT",
            },
            {
                "qa_id": "a",
                "source_id": "sa",
                "gold_answer": "yes",
                "example_state": "NO_ACTION",
            },
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "qa_id": f"q{index}",
                "source_id": f"s{index}",
                "question_group": "audio",
                "expert_name": "muq",
                "expert_available": True,
                "expert_correct": True,
                "expert_support": True,
                "pair_state": "REPAIR",
            }
            for index in range(5)
        ]
    )
    parent = pd.DataFrame(
        [
            {"qa_id": "q0", "parent_answer": "no", "parent_response": "no"},
            {"qa_id": "p", "parent_answer": "yes", "parent_response": "yes"},
            {"qa_id": "a", "parent_answer": "yes", "parent_response": "yes"},
        ]
    )
    effects = pd.DataFrame(
        [{
            "qa_id": "q0",
            "expert_name": "muq",
            "target_mask": 0.8,
            "target_replace": 0.7,
            "control_mask": 0.2,
            "control_replace": 0.1,
            "donor_mask": 0.1,
            "donor_replace": 0.2,
        }]
    )
    details = pd.DataFrame(
        [{"qa_id": "q0", "expert_name": "muq", "detail_target": [1.0, 0.0]}]
    )
    artifacts = prepare_cctdh_artifacts(
        sde_repair_pool=repair,
        example_manifest=examples,
        pair_audit=pairs,
        parent_outputs=parent,
        intervention_effects=effects,
        detail_targets=details,
        n_folds=5,
        seed=18,
        tie_order=["imagebind", "muq", "videomae"],
    )
    validated = artifacts["validated_repair"]
    assert validated.loc[0, "source_id"] == "s0"
    assert validated.loc[0, "selected_expert"] == "muq"
    assert validated.loc[0, "detail_target"] == [1.0, 0.0]
    assert artifacts["parent_protect"]["qa_id"].tolist() == ["p"]
    assert artifacts["parent_anchor"]["qa_id"].tolist() == ["a"]


def test_raw_details_build_source_disjoint_teacher_fit_and_supervision():
    repair = pd.DataFrame([{
        "qa_id": "q0",
        "source_id": "s0",
        "gold_answer": "yes",
        "gold_response": "yes",
        "example_state": "REPAIR",
    }])
    examples = pd.DataFrame([
        repair.iloc[0].to_dict(),
        {
            "qa_id": "p",
            "source_id": "sp",
            "gold_answer": "yes",
            "example_state": "PROTECT",
        },
        {
            "qa_id": "a",
            "source_id": "sa",
            "gold_answer": "yes",
            "example_state": "NO_ACTION",
        },
    ])
    pairs = pd.DataFrame([
        {
            "qa_id": f"q{index}",
            "source_id": f"s{index}",
            "question_group": "audio",
            "expert_name": "muq",
            "expert_available": True,
            "expert_correct": True,
            "expert_support": True,
            "pair_state": "REPAIR",
                "media_path": f"data/{index}.mp4",
            "gold_answer": "yes",
        }
        for index in range(5)
    ])
    parent = pd.DataFrame([
        {"qa_id": "q0", "parent_answer": "no", "parent_response": "no"},
        {"qa_id": "p", "parent_answer": "yes", "parent_response": "yes"},
        {"qa_id": "a", "parent_answer": "yes", "parent_response": "yes"},
    ])
    effects = pd.DataFrame([
        {
            "qa_id": f"q{index}",
            "expert_name": "muq",
            "target_mask": 0.8,
            "target_replace": 0.7,
            "control_mask": 0.2,
            "control_replace": 0.1,
            "donor_mask": 0.1,
            "donor_replace": 0.2,
        }
        for index in range(5)
    ])
    details = pd.DataFrame([
        {
            "qa_id": f"q{index}",
            "expert_name": "muq",
            "target_pooled_feature": [float(index + 1), 1.0, 0.0],
            "control_pooled_feature": [0.0, float(index % 2), 1.0],
        }
        for index in range(5)
    ])
    artifacts = prepare_cctdh_artifacts(
        sde_repair_pool=repair,
        example_manifest=examples,
        pair_audit=pairs,
        parent_outputs=parent,
        intervention_effects=effects,
        detail_targets=None,
        local_details=details,
        latent_dim=2,
        n_folds=5,
        seed=18,
        tie_order=["imagebind", "muq", "videomae"],
    )
    supervision = artifacts["validated_repair"]
    teacher_fit = artifacts["teacher_fit"]
    assert supervision["source_id"].tolist() == ["s0"]
    assert set(teacher_fit["source_id"]).isdisjoint(supervision["source_id"])
    assert teacher_fit["expert_name"].eq("muq").all()
    assert len(supervision.loc[0, "detail_target"]) <= 2
