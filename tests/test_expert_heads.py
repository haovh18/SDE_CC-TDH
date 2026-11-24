import numpy as np
import pytest
import torch

from expert.avqa_4way import (
    AVQACandidateHead,
    _assert_partition_contract,
    _assert_protocol_splits,
    candidate_permutation,
)
from expert.music_avqa_42 import MusicAVQA42Head


def test_music_head_has_paper_42_class_contract():
    model = MusicAVQA42Head(8, 10)
    logits = model(torch.randn(3, 8), torch.randn(3, 10))
    assert logits.shape == (3, 42)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_avqa_shared_scorer_is_candidate_permutation_equivariant():
    model = AVQACandidateHead(8, 10)
    model.eval()
    media = torch.randn(3, 8)
    question = torch.randn(3, 10)
    candidates = torch.randn(3, 4, 10)
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(media, question, candidates)
    permuted = model(media, question, candidates[:, permutation])
    assert original.shape == (3, 4)
    assert torch.allclose(permuted, original[:, permutation], atol=1e-6)
    torch.nn.functional.cross_entropy(original, torch.tensor([0, 1, 2])).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_avqa_training_permutation_is_deterministic_and_complete():
    first = candidate_permutation("qa-1", epoch=3, seed=18)
    second = candidate_permutation("qa-1", epoch=3, seed=18)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == [0, 1, 2, 3]


def test_avqa_protocol_rejects_source_overlap():
    train = [
        {"source_id": f"train-{index}"} for index in range(4050)
    ]
    validation = [
        {"source_id": f"val-{index}"} for index in range(450)
    ]
    _assert_protocol_splits(train, validation)
    validation[0]["source_id"] = "train-0"
    with pytest.raises(ValueError, match="overlap"):
        _assert_protocol_splits(train, validation)


def test_avqa_protocol_uses_paper_partition_counts():
    counts = {
        "expert_train": 4050,
        "expert_internal_val": 450,
        "adapt": 3524,
        "tune": 997,
        "holdout": 997,
    }
    rows = [
        {"partition": partition, "source_id": f"{partition}-{index}"}
        for partition, count in counts.items()
        for index in range(count)
    ]
    _assert_partition_contract(rows)
