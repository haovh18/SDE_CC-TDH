#!/usr/bin/env python3
"""Download the public AVQA and MUSIC-AVQA dataset files."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), dict):
        raise ValueError("dataset configuration must define a datasets mapping")
    return value


def clone_at(url: str, revision: str, target: Path) -> str:
    if not (target / ".git").is_dir():
        placeholder = target / ".gitkeep"
        placeholder.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--all", "--tags"], check=True)
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", revision],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()


def snapshot(
    repo_id: str,
    target: Path,
    *,
    revision: str | None = None,
) -> str:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("install huggingface-hub before downloading datasets") from error
    resolved = str(HfApi().dataset_info(repo_id, revision=revision).sha)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved,
        local_dir=target,
    )
    return resolved


def stream_download(url: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={offset}-"} if offset else {},
    )
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=8 << 20)
    partial.replace(target)


def record(
    name: str,
    source: str,
    revision: str,
    paths: list[Path],
) -> None:
    manifest_path = ROOT / "download_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"schema_version": 1, "datasets": {}}
    )
    manifest["datasets"][name] = {
        "source": source,
        "resolved_revision": revision,
        "paths": [str(path.resolve()) for path in paths],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def download_avqa(config: dict[str, Any]) -> None:
    spec = config["datasets"]["avqa"]
    target = ROOT / spec["local_dir"]
    revision = snapshot(
        spec["repo_id"],
        target,
        revision=spec.get("revision"),
    )
    record("avqa", spec["repo_id"], revision, [target])
    print(f"avqa: {target}")


def download_music(config: dict[str, Any], include_additional_videos: bool) -> None:
    spec = config["datasets"]["music_avqa"]
    repository = ROOT / spec["git_dir"]
    git_revision = clone_at(spec["git_url"], spec["git_revision"], repository)
    annotations = ROOT / spec["hf_annotations_dir"]
    hf_revision = snapshot(spec["hf_annotations_repo"], annotations)
    paths = [repository, annotations]
    if include_additional_videos:
        archive = ROOT / spec["archive_dir"]
        stream_download(spec["additional_video_archive"], archive)
        paths.append(archive)
    record(
        "music_avqa",
        f"{spec['git_url']} + {spec['hf_annotations_repo']}",
        f"git:{git_revision};hf:{hf_revision}",
        paths,
    )
    print(f"music_avqa: {ROOT / 'Music_AVQA'}")
    print("Place the authorized original media under dataset/Music_AVQA/media.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=("music_avqa", "avqa", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--include-music-additional-videos", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.name in {"avqa", "all"}:
        download_avqa(config)
    if args.name in {"music_avqa", "all"}:
        download_music(config, args.include_music_additional_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
