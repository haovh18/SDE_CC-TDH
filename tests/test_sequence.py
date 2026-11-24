import torch

from sde_cctdh.training import response_nlls


def test_response_nll_uses_only_labeled_tokens_and_normalizes_length():
    logits = torch.zeros(1, 5, 3)
    labels = torch.tensor([[-100, -100, 1, 2, -100]])
    losses = response_nlls(logits, labels)
    assert torch.allclose(losses, torch.tensor([torch.log(torch.tensor(3.0))]))


def test_response_nll_rejects_empty_target():
    logits = torch.zeros(1, 3, 2)
    labels = torch.full((1, 3), -100)
    try:
        response_nlls(logits, labels)
    except ValueError as error:
        assert "supervised token" in str(error)
    else:
        raise AssertionError("empty response target was accepted")
