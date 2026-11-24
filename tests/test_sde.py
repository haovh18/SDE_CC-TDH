import pandas as pd
import torch

from sde_cctdh.sde import SDEWeights, build_sde_pools, compute_sde_objective
from sde_cctdh.training import project_away


def test_sde_uses_separate_pool_means():
    result = compute_sde_objective(
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0]),
        torch.tensor([4.0, 6.0, 8.0]),
        weights=SDEWeights(repair=1.0, protect=2.0, anchor=0.5),
    )
    assert torch.isclose(result["loss"], torch.tensor(2.0 + 4.0 + 3.0))


def test_basis_projection_is_unconditional():
    gradient = torch.tensor([1.0, 2.0])
    basis = torch.tensor([[1.0], [0.0]])
    assert torch.allclose(project_away(gradient, basis), torch.tensor([0.0, 2.0]))


def test_sde_pool_targets_keep_complete_responses():
    examples = pd.DataFrame([
        {"qa_id": "r", "source_id": "1", "gold_answer": "gold", "example_state": "REPAIR"},
        {"qa_id": "p", "source_id": "2", "gold_answer": "base", "example_state": "PROTECT"},
        {"qa_id": "o", "source_id": "3", "gold_answer": "base", "example_state": "NO_ACTION"},
    ])
    cache = pd.DataFrame([
        {"qa_id": "r", "gold_response": "The rendered gold response", "base_response": "raw wrong"},
        {"qa_id": "p", "gold_response": "gold", "base_response": "Original Base Response."},
        {"qa_id": "o", "gold_response": "gold", "base_response": "Original anchor response."},
    ])
    repair, protect, anchor = build_sde_pools(examples, cache)
    assert repair.loc[0, "target_response"] == "The rendered gold response"
    assert protect.loc[0, "target_response"] == "Original Base Response."
    assert anchor.loc[0, "target_response"] == "Original anchor response."
