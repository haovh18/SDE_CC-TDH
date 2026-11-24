#!/usr/bin/env python3
"""Build paper-defined pair and example RPU manifests from JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sde_cctdh.rpu import build_rpu_manifest


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = frame.astype(object).where(pd.notnull(frame), None).to_dict("records")
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-pairs", type=Path, required=True)
    parser.add_argument("--pair-output", type=Path, required=True)
    parser.add_argument("--example-output", type=Path, required=True)
    parser.add_argument(
        "--unavailable-policy",
        choices=["skip", "uncertain"],
        default="skip",
    )
    args = parser.parse_args()
    rows = pd.DataFrame(
        json.loads(line)
        for line in args.expert_pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    pairs, examples = build_rpu_manifest(
        rows,
        unavailable_policy=args.unavailable_policy,
    )
    write_jsonl(args.pair_output, pairs)
    write_jsonl(args.example_output, examples)


if __name__ == "__main__":
    main()

