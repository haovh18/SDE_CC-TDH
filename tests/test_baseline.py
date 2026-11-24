import json
from argparse import Namespace
from pathlib import Path

from baseline.evaluate import ensemble_predictions, score_rows
from baseline.protocol import (
    INVALID,
    load_config,
    normalize_music_answer,
    parse_avqa_letter,
    render_prompt,
)
from dataset.prepare_dataset import prepare_dataset


def test_frozen_prompts_and_parsers():
    config = load_config()
    assert config["seed"] == 18
    assert set(config["distillation"]) == {"vanilla_kd", "rwkd", "ufa"}
    assert config["distillation"]["ufa"]["hidden_layer"] == 30
    music = {
        "dataset": "music_avqa",
        "qa_id": "m1",
        "source_id": "s1",
        "media_path": "media/unused.mp4",
        "question": "Which instrument is sounding?",
    }
    prompt = render_prompt(music, config)
    assert "accordion, acoustic_guitar" in prompt
    assert prompt.endswith("Which instrument is sounding?")
    vocabulary = config["protocols"]["music_avqa_42"]["answer_vocabulary"]
    assert normalize_music_answer(" Electric Bass. ", vocabulary) == "electric_bass"
    assert normalize_music_answer("explanation: piano", vocabulary) == INVALID
    assert parse_avqa_letter("(b)") == "B"
    assert parse_avqa_letter("The answer is B") == INVALID


def test_prepare_avqa_builds_baseline_and_expert_manifests(tmp_path: Path):
    media = tmp_path / "videos"
    media.mkdir()
    (media / "abc_10.mp4").write_bytes(b"video")
    annotation = tmp_path / "qa.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "video_name": "abc_10",
                    "question_text": "What happens?",
                    "multi_choice": ["a", "b", "c", "d"],
                    "answer": 2,
                }
            ]
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol"
    summary = prepare_dataset(
        Namespace(
            dataset="avqa",
            input=None,
            annotation=annotation,
            media_root=tmp_path,
            protocol_root=protocol,
            partition_field="partition",
            default_partition="holdout",
        )
    )
    inference = protocol / "manifests/inference.jsonl"
    labels = protocol / "manifests/labels.jsonl"
    infer_row = json.loads(inference.read_text())
    label_row = json.loads(labels.read_text())
    full_row = json.loads(
        (protocol / "data/full_qa_manifest.json").read_text()
    )[0]
    assert "gold_index" not in infer_row
    assert infer_row["source_id"] == "abc"
    assert label_row["gold_index"] == 2
    assert full_row["partition"] == "holdout"
    assert (protocol / "data/full_media_manifest.csv").is_file()
    assert summary["qa_rows"] == 1


def test_expert_ensemble_uses_input_order_for_ties():
    config = load_config()
    labels = [{"dataset": "avqa", "qa_id": "q", "source_id": "s", "gold_index": 0}]
    inputs = [
        ("imagebind", {"q": "A"}),
        ("muq", {"q": "B"}),
        ("videomae", {"q": "C"}),
    ]
    ensemble = ensemble_predictions(labels, inputs)
    assert ensemble[0]["normalized_prediction"] == "A"
    metrics, _ = score_rows(labels, {"q": "A"}, config)
    assert metrics["accuracy"] == 1.0
