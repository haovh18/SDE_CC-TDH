#!/usr/bin/env python3
"""Reject an unresolved config before a formal model-specific launch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sde_cctdh.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--require-resolved", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, require_resolved=args.require_resolved)
    print(json.dumps({
        "status": "PASS",
        "method": config["method"],
        "config": str(args.config),
    }, indent=2))


if __name__ == "__main__":
    main()

