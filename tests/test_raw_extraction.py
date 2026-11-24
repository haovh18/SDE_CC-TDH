import argparse
import csv
import json

import numpy as np
import torch

from expert.extract_raw_features import _l2, _options, _question, pack_music
from expert.local_validation import (
    SUPPORTED_EXPERTS,
    _common_bins,
    _imagebind_artifacts,
    _imagebind_replacement,
)


def test_l2_normalizes_rows_and_preserves_zero_rows():
    values = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = _l2(values)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_avqa_options_accept_native_multi_choice():
    assert _options({"qa_id": "q1", "multi_choice": ["a", "b", "c", "d"]}) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_music_raw_question_template_is_materialized():
    assert (
        _question(
            {
                "question_content": "Is the <instrument> playing <location>?",
                "templ_values": ["piano", "indoors"],
            }
        )
        == "Is the piano playing indoors?"
    )


def test_imagebind_is_a_local_handoff_expert():
    assert "imagebind" in SUPPORTED_EXPERTS
    target, control, windows = _common_bins(
        {
            "vision": torch.tensor([8.0, 7.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0]),
            "audio": torch.tensor([6.0, 5.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        },
        0.25,
    )
    assert (target, control) == (0, 2)
    assert windows["vision"][target].tolist() == [0, 1]

    donor = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    selected = torch.zeros(3, 4, 3)
    replacement = _imagebind_replacement(
        donor,
        torch.tensor([1, 2, 3, 4]),
        selected,
    )
    assert replacement.shape == selected.shape


def test_imagebind_local_artifacts_match_cctdh_schema():
    class Encoder:
        hook_layer = 11

        @staticmethod
        def fused_feature(vision, audio, _payload):
            value = vision.mean(dim=(0, 1)) + audio.mean(dim=(0, 1))
            return torch.nn.functional.normalize(value, dim=0)

    class Head(torch.nn.Module):
        def forward(self, media, _question):
            logits = torch.zeros(media.shape[0], 42)
            logits[:, 0] = 3.0 + media[:, 0]
            return logits

    generator = torch.Generator().manual_seed(18)
    current = {
        "vision_hidden": torch.randn(8, 1, 4, generator=generator),
        "audio_hidden": torch.randn(8, 1, 4, generator=generator),
        "vision_reduce_shape": (1, 1),
        "audio_reduce_shape": (1, 1),
    }
    donor = {
        "vision_hidden": torch.randn(8, 1, 4, generator=generator),
        "audio_hidden": torch.randn(8, 1, 4, generator=generator),
        "vision_reduce_shape": (1, 1),
        "audio_reduce_shape": (1, 1),
    }
    effects, details = _imagebind_artifacts(
        encoder=Encoder(),
        current=current,
        donor=donor,
        head=Head(),
        normalizer={
            "media_mean": np.zeros(4, dtype=np.float32),
            "media_std": np.ones(4, dtype=np.float32),
        },
        question=torch.zeros(1, 2),
        gold_index=0,
        window_fraction=0.25,
        device=torch.device("cpu"),
    )
    for role in ("target", "control", "donor"):
        for operator in ("mask", "replace"):
            assert np.isfinite(effects[f"{role}_{operator}"])
    assert len(details["target_pooled_feature"]) == 8
    assert len(details["control_pooled_feature"]) == 8


def test_pack_music_builds_all_three_head_inputs(tmp_path):
    data = tmp_path / "data"
    text = tmp_path / "features/text"
    imagebind = tmp_path / "features/imagebind"
    other = tmp_path / "features/muq_videomae"
    for directory in (data, text, imagebind, other):
        directory.mkdir(parents=True)

    rows = []
    media_rows = []
    for index in range(44):
        split = "train" if index < 42 else ("val" if index == 42 else "test")
        answer = f"answer_{index if index < 42 else 0:02d}"
        rows.append(
            {
                "qa_id": f"q{index}",
                "video_id": f"v{index}",
                "source_id": f"s{index}",
                "question": f"question {index}",
                "gold_answer": answer,
                "split": split,
            }
        )
        media = tmp_path / f"v{index}.mp4"
        media.write_bytes(b"x")
        media_rows.append({"video_id": f"v{index}", "media_path": str(media)})

    (data / "full_qa_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    with (data / "full_media_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "media_path"])
        writer.writeheader()
        writer.writerows(media_rows)

    (text / "qa_ids.json").write_text(
        json.dumps([row["qa_id"] for row in rows]), encoding="utf-8"
    )
    np.save(text / "imagebind_question.npy", np.ones((44, 5), dtype=np.float16))
    np.save(imagebind / "fused.npy", np.ones((44, 7), dtype=np.float16))
    np.save(other / "muq.npy", np.ones((44, 8), dtype=np.float16))
    np.save(other / "videomae.npy", np.ones((44, 9), dtype=np.float16))

    pack_music(argparse.Namespace(protocol_root=tmp_path))

    output = tmp_path / "features/music_arrays"
    for expert, media_dim in (("imagebind", 7), ("muq", 8), ("videomae", 9)):
        with np.load(output / f"{expert}_arrays.npz") as values:
            assert values["media"].shape == (44, media_dim)
            assert values["question"].shape == (44, 5)
            assert values["train_idx"].tolist() == list(range(42))
            assert values["val_idx"].tolist() == [42]
            assert values["test_idx"].tolist() == [43]
            assert len(values["vocab"]) == 42
    assert len((output / "music_manifest.jsonl").read_text().splitlines()) == 44
