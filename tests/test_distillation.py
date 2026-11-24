import argparse
from pathlib import Path

import numpy as np
import torch

from baseline.distillation import (
    UFAProjectors,
    kd_loss,
    kd_score_gradient,
    prepare,
    unsigned_alignment_loss,
)
from baseline.protocol import load_config, write_jsonl


def test_kd_applies_temperature_to_teacher_distribution():
    temperature = 2.0
    scores = torch.tensor([0.4, -0.3, 0.1])
    teachers = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float32
    )
    teacher_t = teachers.pow(1.0 / temperature)
    teacher_t = teacher_t / teacher_t.sum(dim=1, keepdim=True)
    log_student_t = torch.nn.functional.log_softmax(scores / temperature, dim=0)
    expected_loss = torch.nn.functional.kl_div(
        log_student_t,
        teacher_t.mean(dim=0),
        reduction="sum",
    ) * temperature**2
    expected_gradient = temperature * (
        log_student_t.exp() - teacher_t.mean(dim=0)
    )
    torch.testing.assert_close(
        kd_loss(scores, teachers, temperature=temperature),
        expected_loss,
    )
    torch.testing.assert_close(
        kd_score_gradient(scores, teachers, temperature=temperature),
        expected_gradient,
    )


def test_kd_replay_gradient_matches_autograd():
    scores = torch.tensor([0.4, -0.3, 0.1], requires_grad=True)
    teachers = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float32
    )
    reliability = torch.tensor([0.4, 1.7])
    for weights in (None, reliability):
        loss = kd_loss(
            scores,
            teachers,
            temperature=2.0,
            reliability=weights,
        )
        expected = torch.autograd.grad(loss, scores, retain_graph=True)[0]
        actual = kd_score_gradient(
            scores.detach(),
            teachers,
            temperature=2.0,
            reliability=weights,
        )
        torch.testing.assert_close(actual, expected)


def test_unsigned_alignment_averages_available_pairs_and_updates_expert_projectors():
    projectors = UFAProjectors(
        hidden_dim=5,
        expert_dims={"imagebind": 4, "muq": 3},
        rank=2,
    )
    hidden = torch.randn(4, 5, requires_grad=True)
    embeddings = {
        "imagebind": torch.randn(4, 4),
        "muq": torch.randn(4, 3),
    }
    available = {
        "imagebind": torch.ones(4, dtype=torch.bool),
        "muq": torch.tensor([True, True, False, True]),
    }
    loss = unsigned_alignment_loss(
        hidden,
        embeddings,
        available,
        projectors,
    )
    expected_terms = []
    for expert, values in embeddings.items():
        chosen = torch.nonzero(available[expert], as_tuple=False).flatten()
        expected_terms.append(
            1
            - torch.nn.functional.cosine_similarity(
                projectors.experts[expert](values[chosen]),
                hidden[chosen],
                dim=-1,
            )
        )
    expected = torch.cat(expected_terms).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert not hasattr(projectors, "student")
    assert all(
        isinstance(projector, torch.nn.Sequential)
        and projector[0].out_features == 2
        and projector[1].in_features == 2
        and projector[1].out_features == 5
        and projector[0].bias is None
        and projector[1].bias is None
        for projector in projectors.experts.values()
    )
    assert projectors.experts["imagebind"](embeddings["imagebind"]).shape == (4, 5)
    assert projectors.experts["muq"](embeddings["muq"]).shape == (4, 5)
    assert all(parameter.grad is not None for parameter in projectors.parameters())


def test_prepare_builds_all_three_distillation_inputs(tmp_path: Path):
    vocabulary = load_config()["protocols"]["music_avqa_42"][
        "answer_vocabulary"
    ]
    manifest_rows = []
    label_rows = []
    expert_rows = []
    row_count = 12
    for index in range(row_count):
        qa_id = f"q{index}"
        source_id = f"source-{index}"
        manifest_rows.append(
            {
                "dataset": "music_avqa",
                "qa_id": qa_id,
                "source_id": source_id,
                "media_path": f"media/{index}.mp4",
                "question": f"question {index}",
                "question_group": f"group-{index % 2}",
            }
        )
        label_rows.append(
            {
                "dataset": "music_avqa",
                "qa_id": qa_id,
                "source_id": source_id,
                "gold_answer": vocabulary[index % len(vocabulary)],
            }
        )
        expert_rows.append({"qa_id": qa_id, "source_id": source_id})
    manifest = tmp_path / "manifest.jsonl"
    labels = tmp_path / "labels.jsonl"
    expert_manifest = tmp_path / "expert_manifest.jsonl"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(labels, label_rows)
    write_jsonl(expert_manifest, expert_rows)

    probability_args = []
    array_args = []
    generator = np.random.default_rng(18)
    for expert_index, expert in enumerate(("imagebind", "muq", "videomae")):
        values = generator.random((row_count, len(vocabulary)), dtype=np.float32)
        values /= values.sum(axis=1, keepdims=True)
        probability_path = tmp_path / f"{expert}_probabilities.npz"
        np.savez_compressed(
            probability_path,
            probabilities=values,
            row_indices=np.arange(row_count),
            vocab=np.asarray(vocabulary),
        )
        arrays_path = tmp_path / f"{expert}_arrays.npz"
        np.savez_compressed(
            arrays_path,
            media=generator.normal(size=(row_count, 5 + expert_index)).astype(
                np.float32
            ),
        )
        probability_args.append(f"{expert}={probability_path}")
        array_args.append(f"{expert}={arrays_path}")

    output = tmp_path / "bundle"
    summary = prepare(
        argparse.Namespace(
            seed=18,
            protocol_config=Path("baseline/config.yaml"),
            protocol_name="music_avqa_42",
            manifest=manifest,
            labels=labels,
            expert_manifest=expert_manifest,
            expert_probabilities=probability_args,
            expert_arrays=array_args,
            folds=2,
            sde_log=None,
            output=output,
        )
    )
    assert summary["rows"] == row_count
    assert summary["seed"] == 18
    with np.load(output / "teacher_distributions.npz") as payload:
        assert payload["probabilities"].shape == (row_count, 3, 42)
    with np.load(output / "expert_embeddings.npz") as payload:
        assert payload["imagebind_embeddings"].shape == (row_count, 5)
    assert len((output / "rwkd_weights.jsonl").read_text().splitlines()) == 36
