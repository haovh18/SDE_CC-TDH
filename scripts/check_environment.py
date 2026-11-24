#!/usr/bin/env python3
"""Check one runtime environment and its required local assets."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path


PACKAGES = {
    "qwen": {
        "torch": "2.5.1",
        "transformers": "4.51.3",
        "qwen-omni-utils": "0.0.9",
    },
    "imagebind": {
        "torch": "2.0.1",
        "torchaudio": "2.0.2",
        "pytorchvideo": "0.1.5",
    },
    "temporal": {
        "torch": "2.0.1",
        "torchaudio": "2.0.2",
        "transformers": "4.46.3",
        "muq": "0.1.0",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(PACKAGES))
    parser.add_argument("--asset", type=Path, action="append", default=[])
    args = parser.parse_args()
    package_status = {}
    ok = sys.version_info[:2] == (3, 10)
    for package, expected in PACKAGES[args.profile].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        matched = actual is not None and actual.split("+", 1)[0] == expected
        package_status[package] = {
            "expected": expected,
            "actual": actual,
            "matched": matched,
        }
        ok = ok and matched
    assets = {
        str(path): path.expanduser().exists()
        for path in args.asset
    }
    ok = ok and all(assets.values())
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    ok = ok and all(tools.values())
    result = {
        "status": "PASS" if ok else "FAIL",
        "profile": args.profile,
        "python": sys.version.split()[0],
        "python_expected": "3.10",
        "packages": package_status,
        "system_tools": tools,
        "assets": assets,
    }
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
