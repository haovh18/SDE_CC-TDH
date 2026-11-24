#!/usr/bin/env python3
"""Run the MiniMax-M3 video API baseline without persisting API secrets."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.protocol import (
    append_jsonl,
    load_config,
    parse_prediction,
    read_jsonl,
    render_prompt,
    validate_row,
)


MESSAGES_URL = "https://api.minimaxi.com/anthropic/v1/messages"
FILES_URL = "https://api.minimaxi.com/v1/files/upload"
MODEL_ID = "MiniMax-M3"
INLINE_BYTES = 45 * 1024 * 1024


def redact(value: str | None, key: str) -> str | None:
    if value is None:
        return None
    text = str(value).replace(key, "[REDACTED_API_KEY]")
    return re.sub(
        r"(?i)(authorization|x-api-key)\s*[:=]\s*[^,\s;]+",
        r"\1=[REDACTED]",
        text,
    )[:6000]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mime(path: Path) -> str:
    known = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }
    return known.get(path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")


def response_text(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    value = "".join(
        str(block.get("text", ""))
        for block in payload.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return value or None


class MiniMaxClient:
    def __init__(self, key: str, uploads_path: Path, rpm: float, max_retries: int) -> None:
        import requests

        self.requests = requests
        self.key = key
        self.uploads_path = uploads_path
        self.interval = 60.0 / rpm
        self.max_retries = max_retries
        self.last_request = 0.0
        self.uploads: dict[str, str] = {}
        if uploads_path.exists():
            for row in read_jsonl(uploads_path):
                if row.get("sha256") and row.get("file_id"):
                    self.uploads[str(row["sha256"])] = str(row["file_id"])

    def headers(self, *, json_body: bool) -> dict[str, str]:
        headers = {
            "Authorization": "Bearer " + self.key,
            "X-Api-Key": self.key,
            "anthropic-version": "2023-06-01",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def wait(self) -> None:
        delay = self.interval - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        self.last_request = time.monotonic()

    def post(self, url: str, **kwargs: Any) -> tuple[dict[str, Any] | None, int | None, str | None, int]:
        last_error: str | None = None
        status: int | None = None
        for attempt in range(self.max_retries + 1):
            self.wait()
            try:
                for file_value in kwargs.get("files", {}).values():
                    if isinstance(file_value, tuple) and hasattr(file_value[1], "seek"):
                        file_value[1].seek(0)
                response = self.requests.post(url, timeout=(20, 900), **kwargs)
                status = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if response.ok and isinstance(payload, dict):
                    return payload, status, None, attempt
                last_error = redact(f"HTTP {status}: {response.text[:1000]}", self.key)
                if status != 429 and not (500 <= status <= 599):
                    break
            except self.requests.RequestException as exc:
                last_error = redact(f"{type(exc).__name__}: {exc}", self.key)
            if attempt < self.max_retries:
                time.sleep(min(60.0, 2.0**attempt))
        return None, status, last_error, self.max_retries

    def upload(self, path: Path, digest: str) -> str:
        if digest in self.uploads:
            return self.uploads[digest]
        with path.open("rb") as handle:
            payload, _, error, _ = self.post(
                FILES_URL,
                headers=self.headers(json_body=False),
                data={"purpose": "video_understanding"},
                files={"file": (path.name, handle, mime(path))},
            )
        if error or not payload:
            raise RuntimeError(f"MiniMax file upload failed: {error}")
        file_id = payload.get("file", {}).get("file_id") or payload.get("file_id")
        if not file_id:
            raise RuntimeError("MiniMax upload response has no file_id")
        self.uploads[digest] = str(file_id)
        append_jsonl(self.uploads_path, {"sha256": digest, "file_id": str(file_id), "size": path.stat().st_size})
        return str(file_id)

    def video_block(self, path: Path) -> tuple[dict[str, Any], str]:
        digest = sha256_file(path)
        if path.stat().st_size <= INLINE_BYTES:
            source = {
                "type": "base64",
                "media_type": mime(path),
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                "detail": "default",
                "fps": 1,
            }
            return {"type": "video", "source": source}, "base64"
        file_id = self.upload(path, digest)
        source = {"type": "url", "url": "mm_file://" + file_id, "detail": "default", "fps": 1}
        return {"type": "video", "source": source}, "mm_file"

    def ask(
        self,
        row: dict[str, Any],
        prompt: str,
        system: str | None,
        *,
        prior_text: str | None = None,
        correction_prompt: str | None = None,
    ) -> tuple[dict[str, Any] | None, int | None, str | None, int, str]:
        video, transport = self.video_block(Path(row["media_path"]))
        first = {"role": "user", "content": [video, {"type": "text", "text": prompt}]}
        messages = [first]
        if correction_prompt is not None:
            messages.extend(
                [
                    {"role": "assistant", "content": [{"type": "text", "text": prior_text or ""}]},
                    {"role": "user", "content": [{"type": "text", "text": correction_prompt}]},
                ]
            )
        payload: dict[str, Any] = {
            "model": MODEL_ID,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 8,
            "stream": False,
            "service_tier": "standard",
            "messages": messages,
        }
        if system:
            payload["system"] = system
        data, status, error, retries = self.post(
            MESSAGES_URL, headers=self.headers(json_body=True), json=payload
        )
        return data, status, error, retries, transport


def minimax_prompt(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, str | None, str | None]:
    if row["dataset"] == "music_avqa":
        return render_prompt(row, config), None, None
    protocol = config["protocols"]["avqa_4way"]
    prompt = protocol["minimax_user_prompt"].format(
        question=row["question"],
        option_a=row["options"][0],
        option_b=row["options"][1],
        option_c=row["options"][2],
        option_d=row["options"][3],
    )
    return prompt, protocol["minimax_system_prompt"], protocol["minimax_format_retry"]


def run(args: argparse.Namespace) -> None:
    key = "".join(os.environ.get("MINIMAX_API_KEY", "").split())
    if not key:
        raise RuntimeError("MINIMAX_API_KEY is required and is never written to disk")
    config = load_config(args.config)
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        validate_row(row)
    existing = read_jsonl(args.output) if args.output.exists() else []
    terminal_states = {"success", "invalid_unparsed"}
    completed = {str(row["qa_id"]) for row in existing if row.get("terminal_status") in terminal_states}
    client = MiniMaxClient(key, args.output.with_suffix(".uploads.jsonl"), args.rpm, args.max_retries)
    for position, row in enumerate(rows, 1):
        if str(row["qa_id"]) in completed:
            continue
        started = time.monotonic()
        prompt, system, correction = minimax_prompt(row, config)
        data, status, error, retries, transport = client.ask(row, prompt, system)
        raw = response_text(data)
        prediction = parse_prediction(raw, row["dataset"], config)
        format_retry = False
        if status and 200 <= status < 300 and prediction == "__invalid__" and correction:
            format_retry = True
            data, status, error, extra_retries, transport = client.ask(
                row, prompt, system, prior_text=raw, correction_prompt=correction
            )
            retries += extra_retries
            raw = response_text(data)
            prediction = parse_prediction(raw, row["dataset"], config)
        model_id = (data or {}).get("model")
        success = bool(status and 200 <= status < 300 and model_id == MODEL_ID)
        if not success:
            prediction = "__invalid__"
            if model_id and model_id != MODEL_ID:
                error = f"model_id_mismatch: {model_id}"
        terminal_status = "success" if success and prediction != "__invalid__" else "invalid_unparsed" if success else "failure"
        record = {
            "qa_id": str(row["qa_id"]),
            "source_id": str(row["source_id"]),
            "dataset": row["dataset"],
            "model": "minimax_m3",
            "api_model_id": model_id,
            "raw_output": raw,
            "normalized_prediction": prediction,
            "terminal_status": terminal_status,
            "http_status": status,
            "media_transport": transport,
            "format_retry": format_retry,
            "retries": retries,
            "usage": (data or {}).get("usage", {}),
            "error": error,
            "elapsed_seconds": time.monotonic() - started,
        }
        append_jsonl(args.output, record)
        print(
            f"[{position}/{len(rows)}] {row['qa_id']} {record['terminal_status']} "
            f"{record['normalized_prediction']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpm", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
