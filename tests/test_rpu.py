import pandas as pd

from sde_cctdh.rpu import build_rpu_manifest


def test_pair_truth_table_and_example_aggregation():
    rows = pd.DataFrame([
        {"qa_id": "r", "source_id": "1", "base_answer": "a", "gold_answer": "b", "expert_name": "e1", "expert_answer": "b", "expert_available": True},
        {"qa_id": "r", "source_id": "1", "base_answer": "a", "gold_answer": "b", "expert_name": "e2", "expert_answer": "a", "expert_available": True},
        {"qa_id": "p", "source_id": "2", "base_answer": "b", "gold_answer": "b", "expert_name": "e1", "expert_answer": "a", "expert_available": True},
        {"qa_id": "u", "source_id": "3", "base_answer": "a", "gold_answer": "b", "expert_name": "e1", "expert_answer": "a", "expert_available": True},
        {"qa_id": "o", "source_id": "4", "base_answer": "b", "gold_answer": "b", "expert_name": "e1", "expert_answer": "b", "expert_available": True},
    ])
    pairs, examples = build_rpu_manifest(rows)
    states = examples.set_index("qa_id")["example_state"].to_dict()
    assert states == {"r": "REPAIR", "p": "PROTECT", "u": "UNCERTAIN", "o": "NO_ACTION"}
    assert pairs.loc[(pairs.qa_id == "r") & (pairs.expert_name == "e1"), "pair_state"].item() == "REPAIR"


def test_unavailable_pair_does_not_create_repair():
    rows = pd.DataFrame([
        {"qa_id": "x", "source_id": "1", "base_answer": "a", "gold_answer": "b", "expert_name": "e", "expert_answer": "b", "expert_available": False},
    ])
    pairs, examples = build_rpu_manifest(rows)
    assert pd.isna(pairs.loc[0, "pair_state"])
    assert examples.loc[0, "example_state"] == "UNCERTAIN"

