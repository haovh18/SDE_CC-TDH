"""Shared prompt, manifest, and answer protocol for paper baselines."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import yaml


INVALID = "__invalid__"
LETTERS = ("A", "B", "C", "D")
CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported baseline config")
    vocabulary = value["protocols"]["music_avqa_42"]["answer_vocabulary"]
    if len(vocabulary) != 42 or len(set(vocabulary)) != 42 or not all(isinstance(item, str) for item in vocabulary):
        raise ValueError("MUSIC-AVQA protocol requires 42 unique string answers")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def normalize_music_answer(raw: Any, vocabulary: Iterable[str]) -> str:
    value = unicodedata.normalize("NFKC", str(raw or "")).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = value.rstrip(" .,!?:;。！？：；")
    if value == "electric bass":
        value = "electric_bass"
    return value if value in set(vocabulary) else INVALID


def parse_avqa_letter(raw: Any) -> str:
    value = unicodedata.normalize("NFKC", str(raw or "")).upper().strip()
    value = re.sub(r"\s+", "", value)
    value = value.strip(".,:;!?()[]{}\"'`。，：；！？（）【】《》")
    return value if value in LETTERS else INVALID


def protocol_name(dataset: str) -> str:
    names = {"music_avqa": "music_avqa_42", "avqa": "avqa_4way"}
    try:
        return names[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc


def validate_row(row: dict[str, Any], *, require_media: bool = True) -> None:
    required = ("dataset", "qa_id", "source_id", "media_path", "question")
    missing = [key for key in required if not str(row.get(key, "")).strip()]
    if missing:
        raise ValueError(f"row is missing {missing}: {row.get('qa_id', '<unknown>')}")
    if row["dataset"] == "avqa":
        options = row.get("options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"AVQA row needs four options: {row['qa_id']}")
    elif row["dataset"] != "music_avqa":
        raise ValueError(f"unsupported dataset in {row['qa_id']}: {row['dataset']}")
    if require_media and not Path(row["media_path"]).is_file():
        raise FileNotFoundError(row["media_path"])


def render_prompt(row: dict[str, Any], config: dict[str, Any]) -> str:
    validate_row(row, require_media=False)
    protocol = config["protocols"][protocol_name(row["dataset"])]
    if row["dataset"] == "music_avqa":
        return protocol["prompt"].format(
            answers=", ".join(protocol["answer_vocabulary"]),
            question=row["question"],
        )
    return protocol["prompt"].format(
        question=row["question"],
        option_a=row["options"][0],
        option_b=row["options"][1],
        option_c=row["options"][2],
        option_d=row["options"][3],
    )


def parse_prediction(raw: Any, dataset: str, config: dict[str, Any]) -> str:
    if dataset == "music_avqa":
        vocabulary = config["protocols"]["music_avqa_42"]["answer_vocabulary"]
        return normalize_music_answer(raw, vocabulary)
    if dataset == "avqa":
        return parse_avqa_letter(raw)
    raise ValueError(f"unsupported dataset: {dataset}")


def gold_prediction(label: dict[str, Any], config: dict[str, Any]) -> str:
    dataset = label["dataset"]
    if dataset == "music_avqa":
        return parse_prediction(label["gold_answer"], dataset, config)
    if "gold_index" in label:
        index = int(label["gold_index"])
        if index not in range(4):
            raise ValueError(f"bad gold_index for {label['qa_id']}: {index}")
        return LETTERS[index]
    return parse_avqa_letter(label.get("gold_answer"))
