#!/usr/bin/env python3
"""Download AVLLMs, expert encoders, and external baseline assets."""
from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUTPUT = ROOT


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model configuration must be a mapping")
    return value


def clone_at(url: str, revision: str, target: Path) -> str:
    if not (target / ".git").is_dir():
        (target / ".gitkeep").unlink(missing_ok=True)
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


def stream_download(
    url: str,
    target: Path,
    *,
    expected_sha256: str | None = None,
) -> None:
    if target.is_file() and (
        expected_sha256 is None or file_sha256(target) == expected_sha256
    ):
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
    if expected_sha256 and file_sha256(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {target.name}")
    partial.replace(target)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(
    repo_id: str,
    target: Path,
    *,
    repo_type: str = "model",
    revision: str | None = None,
) -> str:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("install the baseline optional dependencies") from error
    api = HfApi()
    info = (
        api.dataset_info(repo_id, revision=revision)
        if repo_type == "dataset"
        else api.model_info(repo_id, revision=revision)
    )
    resolved = str(info.sha)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=resolved,
        local_dir=target,
    )
    return resolved


def _record(
    output_root: Path,
    category: str,
    name: str,
    source: str,
    revision: str,
    paths: list[Path],
) -> None:
    manifest_path = output_root / "download_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"schema_version": 1, "models": {}}
    )
    key = f"{category}/{name}"
    manifest["models"][key] = {
        "source": source,
        "resolved_revision": revision,
        "paths": [str(path.resolve()) for path in paths],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def download_avllm(
    name: str,
    output_root: Path,
    config: dict[str, Any],
) -> None:
    spec = config["avllms"][name]
    target = output_root / spec["local_dir"]
    revision = _snapshot(
        spec["repo_id"],
        target,
        revision=spec.get("revision"),
    )
    for relative, expected in spec.get("files_sha256", {}).items():
        path = target / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"{name}: frozen asset hash mismatch for {relative}")
    _record(output_root, "avllm", name, spec["repo_id"], revision, [target])
    print(f"{name}: {target}")


def download_expert(
    name: str,
    output_root: Path,
    config: dict[str, Any],
) -> None:
    spec = config["experts"][name]
    if name in {"muq", "videomae"}:
        target = output_root / spec["local_dir"]
        revision = _snapshot(
            spec["repo_id"],
            target,
            revision=spec.get("revision"),
        )
        _record(output_root, "expert", name, spec["repo_id"], revision, [target])
        print(f"{name}: {target}")
        return
    code = output_root / spec["code_dir"]
    revision = clone_at(spec["code_url"], spec["code_revision"], code)
    checkpoint = output_root / spec["checkpoint_dir"]
    stream_download(
        spec["checkpoint_url"],
        checkpoint,
        expected_sha256=spec.get("checkpoint_sha256"),
    )
    _record(
        output_root,
        "expert",
        name,
        spec["code_url"],
        revision,
        [code, checkpoint],
    )
    print(f"{name}: code={code} checkpoint={checkpoint}")


def download_external(
    name: str,
    output_root: Path,
    config: dict[str, Any],
) -> None:
    spec = config["external"][name]
    code = output_root / spec["git_dir"]
    git_revision = clone_at(spec["git_url"], spec["git_revision"], code)
    checkpoint = output_root / spec["checkpoint_dir"]
    hf_revision = _snapshot(
        spec["checkpoint_repo"],
        checkpoint,
        revision=spec["checkpoint_revision"],
    )
    _record(
        output_root,
        "external",
        name,
        f"{spec['git_url']} + {spec['checkpoint_repo']}",
        f"git:{git_revision};hf:{hf_revision}",
        [code, checkpoint],
    )
    print(f"{name}: {code.parent}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)

    avllm = commands.add_parser("avllm")
    avllm.add_argument(
        "name",
        choices=("qwen_omni_3b", "qwen_omni_7b", "minicpm_o_45_gptq", "all"),
    )
    expert = commands.add_parser("expert")
    expert.add_argument("name", choices=("imagebind", "muq", "videomae", "clip", "all"))
    external = commands.add_parser("external")
    external.add_argument("name", choices=("qa_tiger",))
    commands.add_parser("all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output = args.output_root.expanduser().resolve()
    if args.command in {"avllm", "all"}:
        names = (
            tuple(config["avllms"])
            if args.command == "all" or args.name == "all"
            else (args.name,)
        )
        for name in names:
            download_avllm(name, output, config)
    if args.command in {"expert", "all"}:
        names = (
            tuple(config["experts"])
            if args.command == "all" or args.name == "all"
            else (args.name,)
        )
        for name in names:
            download_expert(name, output, config)
    if args.command in {"external", "all"}:
        names = tuple(config["external"]) if args.command == "all" else (args.name,)
        for name in names:
            download_external(name, output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
