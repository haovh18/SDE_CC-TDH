#!/usr/bin/env python3
"""Build shared baseline and expert manifests from native QA annotations."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


MEDIA_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def materialize_music_question(row: dict[str, Any]) -> str:
    rendered = row.get("question_rendered")
    if rendered not in (None, ""):
        return normalize_space(rendered)
    question = str(row.get("question", row.get("question_content", "")))
    values = row.get("templ_values", [])
    if isinstance(values, str):
        values = json.loads(values)
    for value in values or []:
        question = re.sub(r"<[^<>]+>", str(value), question, count=1)
    if re.search(r"<[^<>]+>", question):
        raise ValueError(f"unresolved question placeholder: {question}")
    return normalize_space(question)


def read_annotation(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON array")
        return [dict(row) for row in value]
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported annotation format: {path}")


def parse_partition_input(value: str) -> tuple[str, Path]:
    partition, separator, path = value.partition("=")
    if not separator or not partition or not path:
        raise argparse.ArgumentTypeError("--input must be PARTITION=PATH")
    return partition, Path(path)


def load_partitioned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    def native_partition(row: dict[str, Any]) -> Any:
        if args.partition_field != "auto":
            return row.get(args.partition_field)
        for field in ("partition", "split", "official_split", "experiment_split"):
            value = row.get(field)
            if value not in (None, ""):
                return value
        return None

    if args.input:
        rows = []
        for partition, path in args.input:
            for row in read_annotation(path):
                existing = native_partition(row)
                if existing not in (None, "", partition):
                    raise ValueError(
                        f"{path} contains partition {existing!r}, expected {partition!r}"
                    )
                rows.append(row | {"_prepared_partition": partition})
        return rows
    rows = read_annotation(args.annotation)
    prepared = []
    for row in rows:
        partition = native_partition(row)
        if partition in (None, ""):
            partition = args.default_partition
        if partition in (None, ""):
            raise ValueError(
                "row lacks a partition field; pass --default-partition "
                "or use --input PARTITION=PATH"
            )
        prepared.append(row | {"_prepared_partition": str(partition)})
    return prepared


def media_index(media_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for path in media_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        key = path.stem
        if key in index and index[key] != path.resolve():
            ambiguous.add(key)
        else:
            index[key] = path.resolve()
    for key in ambiguous:
        index.pop(key, None)
    return index


def resolve_media(
    row: dict[str, Any],
    media_root: Path,
    index: dict[str, Path],
) -> Path:
    relative = row.get("media_path", row.get("video_path"))
    if relative:
        candidate = Path(str(relative)).expanduser()
        for path in (candidate, media_root / candidate, media_root / "videos" / candidate):
            if path.is_file():
                return path.resolve()
    key = str(
        row.get(
            "clip_id",
            row.get("video_name", row.get("video_id", row.get("sample_id", ""))),
        )
    )
    path = index.get(Path(key).stem)
    if path is None:
        raise FileNotFoundError(f"cannot uniquely resolve media for {key!r}")
    return path


def source_from_clip(clip_id: str) -> str:
    return clip_id.rsplit("_", 1)[0] if "_" in clip_id else clip_id


def _qa_id(row: dict[str, Any], dataset: str, position: int) -> str:
    native = row.get("qa_id", row.get("question_id", row.get("id")))
    return str(native if native is not None else f"{dataset}::{position:08d}")


def _partition(row: dict[str, Any]) -> str:
    return str(row["_prepared_partition"])


def _question_group(row: dict[str, Any]) -> str | None:
    for field in (
        "question_group",
        "question_type",
        "type",
        "template",
        "question_template",
    ):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def prepare_avqa_row(
    row: dict[str, Any],
    position: int,
    media: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clip_id = str(row.get("clip_id", row.get("video_name", row.get("video_id"))))
    if clip_id in {"", "None"}:
        clip_id = media.stem
    source_id = str(row.get("source_id", source_from_clip(clip_id)))
    options = row.get("multi_choice", row.get("options"))
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(f"AVQA row {position} must contain four candidates")
    answer = int(row.get("answer", row.get("gold_index")))
    if answer not in range(4):
        raise ValueError(f"AVQA row {position} answer must be in 0..3")
    qa_id = _qa_id(row, "avqa", position)
    question = normalize_space(row.get("question_text", row.get("question", "")))
    full = {
        "qa_id": qa_id,
        "clip_id": clip_id,
        "source_id": source_id,
        "question_text": question,
        "multi_choice": [str(value) for value in options],
        "answer": answer,
        "partition": _partition(row),
    }
    question_group = _question_group(row)
    if question_group is not None:
        full["question_group"] = question_group
    inference = {
        "dataset": "avqa",
        "qa_id": qa_id,
        "source_id": source_id,
        "media_path": str(media),
        "question": question,
        "options": full["multi_choice"],
        "partition": _partition(row),
    }
    if "question_group" in full:
        inference["question_group"] = full["question_group"]
    label = {
        "dataset": "avqa",
        "qa_id": qa_id,
        "source_id": source_id,
        "gold_index": answer,
        "partition": _partition(row),
    }
    return full, inference, label


def prepare_music_row(
    row: dict[str, Any],
    position: int,
    media: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    video_id = str(row.get("video_id", row.get("clip_id", row.get("sample_id", media.stem))))
    source_id = str(row.get("source_id", video_id))
    qa_id = _qa_id(row, "music_avqa", position)
    question = materialize_music_question(row)
    answer = row.get("gold_answer", row.get("anser", row.get("answer")))
    if answer is None:
        raise ValueError(f"MUSIC row {position} has no gold answer")
    full = {
        "qa_id": qa_id,
        "video_id": video_id,
        "source_id": source_id,
        "question": question,
        "gold_answer": str(answer),
        "split": _partition(row),
    }
    question_group = _question_group(row)
    if question_group is not None:
        full["question_group"] = question_group
    inference = {
        "dataset": "music_avqa",
        "qa_id": qa_id,
        "source_id": source_id,
        "media_path": str(media),
        "question": question,
        "partition": _partition(row),
    }
    if "question_group" in full:
        inference["question_group"] = full["question_group"]
    label = {
        "dataset": "music_avqa",
        "qa_id": qa_id,
        "source_id": source_id,
        "gold_answer": str(answer),
        "partition": _partition(row),
    }
    return full, inference, label


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_rows = load_partitioned_rows(args)
    media_root = args.media_root.expanduser().resolve()
    index = media_index(media_root)
    full_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    media_rows: dict[str, dict[str, str]] = {}
    seen_qa: set[str] = set()
    prepare = prepare_avqa_row if args.dataset == "avqa" else prepare_music_row
    id_field = "clip_id" if args.dataset == "avqa" else "video_id"

    for position, row in enumerate(raw_rows):
        media = resolve_media(row, media_root, index)
        full, inference, label = prepare(row, position, media)
        if full["qa_id"] in seen_qa:
            raise ValueError(f"duplicate qa_id: {full['qa_id']}")
        seen_qa.add(full["qa_id"])
        media_id = str(full[id_field])
        media_record = {
            id_field: media_id,
            "source_id": str(full["source_id"]),
            "media_path": str(media),
        }
        previous = media_rows.get(media_id)
        if previous is not None and previous != media_record:
            raise ValueError(f"media ID {media_id!r} maps to inconsistent records")
        media_rows[media_id] = media_record
        full_rows.append(full)
        inference_rows.append(inference)
        label_rows.append(label)

    source_partitions: dict[str, set[str]] = {}
    for row in full_rows:
        source_partitions.setdefault(str(row["source_id"]), set()).add(
            str(row["partition"] if args.dataset == "avqa" else row["split"])
        )
    leaking = {
        source: sorted(partitions)
        for source, partitions in source_partitions.items()
        if len(partitions) > 1
    }
    if leaking:
        examples = list(leaking.items())[:5]
        raise ValueError(f"source IDs cross partition boundaries: {examples}")

    root = args.protocol_root.expanduser().resolve()
    write_json(root / "data/full_qa_manifest.json", full_rows)
    media_path = root / "data/full_media_manifest.csv"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = media_path.with_name(f".{media_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[id_field, "source_id", "media_path"],
        )
        writer.writeheader()
        writer.writerows(media_rows.values())
    os.replace(temporary, media_path)
    write_jsonl(root / "manifests/inference.jsonl", inference_rows)
    write_jsonl(root / "manifests/labels.jsonl", label_rows)
    partitions = sorted(
        {
            row["partition"] if args.dataset == "avqa" else row["split"]
            for row in full_rows
        }
    )
    for partition in partitions:
        selected = [
            index
            for index, row in enumerate(full_rows)
            if (row["partition"] if args.dataset == "avqa" else row["split"])
            == partition
        ]
        write_jsonl(
            root / "manifests" / partition / "inference.jsonl",
            [inference_rows[index] for index in selected],
        )
        write_jsonl(
            root / "manifests" / partition / "labels.jsonl",
            [label_rows[index] for index in selected],
        )
    summary = {
        "dataset": args.dataset,
        "qa_rows": len(full_rows),
        "media_rows": len(media_rows),
        "source_rows": len(source_partitions),
        "partitions": {
            partition: sum(
                1
                for row in full_rows
                if (row["partition"] if args.dataset == "avqa" else row["split"])
                == partition
            )
            for partition in partitions
        },
    }
    write_json(root / "data/preparation_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("avqa", "music_avqa"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--annotation", type=Path)
    sources.add_argument(
        "--input",
        type=parse_partition_input,
        action="append",
        metavar="PARTITION=PATH",
    )
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--partition-field", default="auto")
    parser.add_argument("--default-partition")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = config["datasets"][args.dataset]
    if args.media_root is None:
        args.media_root = ROOT / spec["media_dir"]
    if args.protocol_root is None:
        args.protocol_root = ROOT / spec["protocol_dir"]
    summary = prepare_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
