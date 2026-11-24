import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from dataset.download_datasets import load_config as load_dataset_config
from dataset.prepare_dataset import prepare_dataset
from model.download_models import load_config as load_model_config


def test_download_configs_cover_models_experts_and_datasets():
    model_config = load_model_config(Path("model/config.yaml"))
    dataset_config = load_dataset_config(Path("dataset/config.yaml"))
    assert set(model_config["avllms"]) == {
        "qwen_omni_3b",
        "qwen_omni_7b",
        "minicpm_o_45_gptq",
    }
    assert set(model_config["experts"]) == {
        "imagebind",
        "muq",
        "videomae",
        "clip",
    }
    assert set(dataset_config["datasets"]) == {"avqa", "music_avqa"}
    assert "downloads" not in yaml.safe_load(
        Path("baseline/config.yaml").read_text()
    )


def test_download_targets_stay_inside_dataset_and_model_directories():
    dataset_config = load_dataset_config(Path("dataset/config.yaml"))["datasets"]
    assert (Path("dataset") / dataset_config["avqa"]["local_dir"]).is_dir()
    music = dataset_config["music_avqa"]
    assert (Path("dataset") / music["git_dir"]).is_dir()
    assert (Path("dataset") / music["hf_annotations_dir"]).is_dir()
    assert (Path("dataset") / music["archive_dir"]).parent.is_dir()
    for dataset_name in ("Music_AVQA", "AVQA"):
        root = Path("dataset") / dataset_name
        assert (root / "media").is_dir()
        assert (root / "protocol").is_dir()

    model_config = load_model_config(Path("model/config.yaml"))
    for spec in model_config["avllms"].values():
        assert (Path("model") / spec["local_dir"]).is_dir()
    for spec in model_config["experts"].values():
        for key in ("local_dir", "code_dir", "checkpoint_dir"):
            if key not in spec:
                continue
            target = Path("model") / spec[key]
            assert target.is_dir() if key != "checkpoint_dir" else target.parent.is_dir()
    for spec in model_config["external"].values():
        assert (Path("model") / spec["git_dir"]).is_dir()
        assert (Path("model") / spec["checkpoint_dir"]).is_dir()

    baseline = yaml.safe_load(Path("baseline/config.yaml").read_text())
    for spec in baseline["models"].values():
        if "local_dir" in spec:
            assert Path(spec["local_dir"]).parts[0] == "model"


def test_prepare_music_materializes_questions_and_partition_inputs(tmp_path):
    media = tmp_path / "videos"
    media.mkdir()
    (media / "v1.mp4").write_bytes(b"video")
    annotation = tmp_path / "train.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_id": "m1",
                    "video_id": "v1",
                    "source_id": "source-1",
                    "question_content": "Is <instrument> sounding?",
                    "templ_values": ["piano"],
                    "anser": "yes",
                }
            ]
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol"
    prepare_dataset(
        Namespace(
            dataset="music_avqa",
            input=[("train", annotation)],
            annotation=None,
            media_root=media,
            protocol_root=protocol,
            partition_field="partition",
            default_partition=None,
        )
    )
    full = json.loads(
        (protocol / "data/full_qa_manifest.json").read_text()
    )
    assert full[0]["question"] == "Is piano sounding?"
    assert full[0]["split"] == "train"
    assert (protocol / "manifests/inference.jsonl").is_file()
    assert (protocol / "manifests/labels.jsonl").is_file()
    inference = [
        json.loads(line)
        for line in (
            protocol / "manifests/train/inference.jsonl"
        ).read_text().splitlines()
    ]
    assert inference[0]["partition"] == "train"
    assert (protocol / "manifests/train/labels.jsonl").is_file()


def test_prepare_rejects_source_crossing_partitions(tmp_path):
    media = tmp_path / "videos"
    media.mkdir()
    for name in ("a", "b"):
        (media / f"{name}.mp4").write_bytes(b"video")
    annotation = tmp_path / "qa.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "qa_id": "a",
                    "video_id": "a",
                    "source_id": "same",
                    "question": "q",
                    "gold_answer": "yes",
                    "partition": "train",
                },
                {
                    "qa_id": "b",
                    "video_id": "b",
                    "source_id": "same",
                    "question": "q",
                    "gold_answer": "yes",
                    "partition": "test",
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cross partition"):
        prepare_dataset(
            Namespace(
                dataset="music_avqa",
                input=None,
                annotation=annotation,
                media_root=media,
                protocol_root=tmp_path / "protocol",
                partition_field="partition",
                default_partition=None,
            )
        )
