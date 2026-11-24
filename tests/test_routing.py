import pandas as pd

from sde_cctdh.routing import crossfit_group_reliability, route_validated_experts


def test_router_uses_oof_group_reliability_and_fixed_tie():
    rows = []
    for source in range(8):
        for expert in ("a", "b"):
            rows.append({
                "qa_id": f"q{source}_{expert}",
                "source_id": f"s{source}",
                "expert_name": expert,
                "question_group": "g",
                "expert_available": True,
                "expert_correct": expert == "a",
            })
    audit = crossfit_group_reliability(pd.DataFrame(rows), n_folds=2, seed=18)
    qa = "q0_a"
    candidates = pd.DataFrame([
        {"qa_id": qa, "source_id": "s0", "question_group": "g", "expert_name": "a", "expert_support": True, "kappa": True},
        {"qa_id": qa, "source_id": "s0", "question_group": "g", "expert_name": "b", "expert_support": True, "kappa": True},
    ])
    rel = audit[audit.qa_id.isin(["q0_a", "q0_b"])].copy()
    rel["qa_id"] = qa
    selected = route_validated_experts(candidates, rel, tie_order=["a", "b"])
    assert selected.loc[0, "selected_expert"] == "a"
