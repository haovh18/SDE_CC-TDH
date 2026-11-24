import pandas as pd

from sde_cctdh.metrics import evaluate_predictions


def test_paper_metric_denominators():
    rows = pd.DataFrame([
        {"base_answer": "a", "method_answer": "b", "gold_answer": "b", "example_state": "REPAIR"},
        {"base_answer": "a", "method_answer": "a", "gold_answer": "b", "example_state": "UNCERTAIN"},
        {"base_answer": "b", "method_answer": "a", "gold_answer": "b", "example_state": "PROTECT"},
        {"base_answer": "b", "method_answer": "b", "gold_answer": "b", "example_state": "NO_ACTION"},
    ])
    result = evaluate_predictions(rows)
    assert result["W2C"] == 1
    assert result["C2W"] == 1
    assert result["BER"] == 0.5
    assert result["RR"] == 1.0
    assert result["PR"] == 0.0
    assert result["UFR"] == 0.0

