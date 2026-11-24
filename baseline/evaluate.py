#!/usr/bin/env python3
"""Evaluate baseline predictions or build the paper expert ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.protocol import (
    INVALID,
    LETTERS,
    gold_prediction,
    load_config,
    parse_prediction,
    read_jsonl,
    write_jsonl,
)


PREDICTION_FIELDS = (
    "normalized_prediction",
    "parsed_letter",
    "parsed_option",
    "prediction",
    "predicted_answer",
    "pred_answer",
)


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"prediction JSON must be an array of objects: {path}")
        return value
    if suffix in {".csv", ".parquet"}:
        import pandas as pd

        frame = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
        return frame.to_dict(orient="records")
    raise ValueError(f"unsupported prediction format: {path.suffix}")


def raw_prediction(row: dict[str, Any], dataset: str) -> Any:
    for field in PREDICTION_FIELDS:
        if field in row and row[field] is not None:
            value = row[field]
            if dataset == "avqa" and isinstance(value, int) and value in range(4):
                return LETTERS[value]
            return value
    raise ValueError(f"prediction row has no recognized prediction field: {row.get('qa_id')}")


def latest_predictions(path: Path, dataset_by_id: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    selected: dict[str, tuple[str, bool]] = {}
    for row in read_prediction_rows(path):
        qa_id = str(row.get("qa_id", row.get("id", "")))
        if qa_id not in dataset_by_id:
            continue
        dataset = dataset_by_id[qa_id]
        prediction = parse_prediction(raw_prediction(row, dataset), dataset, config)
        success = str(row.get("terminal_status", "")).lower() in {"success", "pass", "ok"}
        prior = selected.get(qa_id)
        if prior is None or success or not prior[1]:
            selected[qa_id] = (prediction, success)
    return {qa_id: value[0] for qa_id, value in selected.items()}


def load_labels(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    labels = read_jsonl(path)
    by_id: dict[str, str] = {}
    for row in labels:
        qa_id = str(row["qa_id"])
        if qa_id in by_id:
            raise ValueError(f"duplicate label qa_id: {qa_id}")
        by_id[qa_id] = str(row["dataset"])
        if gold_prediction(row, config) == INVALID:
            raise ValueError(f"invalid gold target: {qa_id}")
    return labels, by_id


def score_rows(
    labels: list[dict[str, Any]],
    predictions: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        qa_id = str(label["qa_id"])
        gold = gold_prediction(label, config)
        prediction = predictions.get(qa_id, INVALID)
        rows.append(
            {
                "qa_id": qa_id,
                "source_id": str(label["source_id"]),
                "dataset": label["dataset"],
                "gold": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "missing": qa_id not in predictions,
                "invalid": prediction == INVALID,
            }
        )
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    metrics = {
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "coverage": sum(not row["missing"] for row in rows) / total if total else None,
        "invalid": sum(row["invalid"] for row in rows),
        "missing": sum(row["missing"] for row in rows),
        "source_count": len({row["source_id"] for row in rows}),
    }
    return metrics, rows


def parse_named_inputs(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        parsed.append((name, Path(path)))
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("duplicate input name")
    return parsed


def ensemble_predictions(
    labels: list[dict[str, Any]],
    inputs: list[tuple[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        qa_id = str(label["qa_id"])
        ordered = [(name, values.get(qa_id, INVALID)) for name, values in inputs]
        counts = Counter(value for _, value in ordered if value != INVALID)
        if counts:
            best_count = max(counts.values())
            winners = {value for value, count in counts.items() if count == best_count}
            prediction = next(value for _, value in ordered if value in winners)
        else:
            prediction = INVALID
        rows.append(
            {
                "qa_id": qa_id,
                "source_id": str(label["source_id"]),
                "dataset": label["dataset"],
                "model": "expert_ensemble",
                "normalized_prediction": prediction,
                "votes": {name: value for name, value in ordered},
                "terminal_status": "success",
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    labels, dataset_by_id = load_labels(args.labels, config)
    if args.command == "score":
        predictions = latest_predictions(args.predictions, dataset_by_id, config)
        metrics, details = score_rows(labels, predictions, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.details:
            write_jsonl(args.details, details)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    named = parse_named_inputs(args.input)
    inputs = [(name, latest_predictions(path, dataset_by_id, config)) for name, path in named]
    ensemble = ensemble_predictions(labels, inputs)
    write_jsonl(args.output, ensemble)
    predictions = {row["qa_id"]: row["normalized_prediction"] for row in ensemble}
    metrics, _ = score_rows(labels, predictions, config)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--labels", required=True, type=Path)
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--details", type=Path)

    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--labels", required=True, type=Path)
    ensemble.add_argument("--input", action="append", required=True, help="NAME=predictions.jsonl; order breaks ties")
    ensemble.add_argument("--output", required=True, type=Path)
    ensemble.add_argument("--metrics", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
