#!/usr/bin/env python3
"""Extract frozen ImageBind, MuQ, VideoMAE, and text features from raw media."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


IMAGEBIND_DIM = 1024
MUQ_DIM = 1024
VIDEOMAE_DIM = 768
NUM_FRAMES = 16
MUQ_SAMPLE_RATE = 24_000
MUQ_SAMPLES = MUQ_SAMPLE_RATE * 10


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _l2(value: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(value, axis=-1, keepdims=True)
    return (value / np.maximum(denominator, 1e-12)).astype(np.float32)


def _load_qa_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "data/full_qa_manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON array")
    return rows


def _load_media_rows(root: Path) -> list[dict[str, str]]:
    path = root / "data/full_media_manifest.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} must contain at least one media row")
    for row in rows:
        _media_id(row)
        media_path = Path(row["media_path"]).expanduser()
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        row["media_path"] = str(media_path.resolve())
    return rows


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    raise KeyError(f"none of the required fields are present: {names}")


def _qa_id(row: dict[str, Any]) -> str:
    return str(_first(row, ("qa_id", "question_id", "id")))


def _media_id(row: dict[str, Any]) -> str:
    return str(_first(row, ("clip_id", "video_id", "sample_id")))


def _question(row: dict[str, Any]) -> str:
    for name in ("question_text", "question_rendered"):
        value = row.get(name)
        if value is not None and value != "":
            return str(value)
    question = str(_first(row, ("question", "question_content")))
    values = row.get("templ_values", [])
    if isinstance(values, str):
        values = json.loads(values)
    for value in values or []:
        question = re.sub(r"<[^<>]+>", str(value), question, count=1)
    if re.search(r"<[^<>]+>", question):
        raise ValueError(f"unresolved question placeholder: {question}")
    return re.sub(r"\s+", " ", question).strip()


def _options(row: dict[str, Any]) -> list[str]:
    values = row.get("multi_choice") or row.get("options")
    if values is None:
        values = [row.get(f"option_{letter}") for letter in "ABCD"]
    if not isinstance(values, list) or len(values) != 4 or any(x is None for x in values):
        raise ValueError(f"QA {_qa_id(row)} does not contain exactly four options")
    return [str(value) for value in values]


def _require_cuda(device: str) -> Any:
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("the frozen expert extraction protocol requires CUDA")
    return torch


def _load_imagebind(
    source: Path,
    checkpoint: Path,
    device: str,
) -> tuple[Any, Any, Any]:
    torch = _require_cuda(device)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(source.resolve()))
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    model = imagebind_model.imagebind_huge(pretrained=False)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    return model, data, ModalityType


def _batched_text(
    values: list[str],
    encode: Callable[[list[str]], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(values), batch_size):
        parts.append(encode(values[start : start + batch_size]))
        print(
            json.dumps(
                {
                    "text_complete": min(start + batch_size, len(values)),
                    "text_total": len(values),
                }
            ),
            flush=True,
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def extract_text(args: argparse.Namespace) -> None:
    torch = _require_cuda(args.device)
    rows = _load_qa_rows(args.protocol_root)
    qa_ids = [_qa_id(row) for row in rows]
    questions = [_question(row) for row in rows]
    options = [_options(row) for row in rows] if args.dataset == "avqa" else []
    model, data, modality = _load_imagebind(
        args.imagebind_source,
        args.imagebind_checkpoint,
        args.device,
    )

    def imagebind_encode(texts: list[str]) -> np.ndarray:
        tokens = data.load_and_transform_text(texts, args.device)
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if args.dataset == "avqa"
            else _null_context()
        )
        with torch.inference_mode(), context:
            result = model({modality.TEXT: tokens})[modality.TEXT]
        value = result.detach().float().cpu().numpy()
        return value if args.dataset == "avqa" else _l2(value)

    ib_batch = 32 if args.dataset == "avqa" else args.text_batch_size
    ib_question = _batched_text(questions, imagebind_encode, ib_batch)
    text_dir = args.protocol_root / "features/text"
    _atomic_npy(
        text_dir / "imagebind_question.npy",
        ib_question.astype(np.float32 if args.dataset == "avqa" else np.float16),
    )
    if args.dataset == "avqa":
        flat_options = [value for group in options for value in group]
        ib_options = _batched_text(flat_options, imagebind_encode, 32).reshape(
            len(rows), 4, -1
        )
        _atomic_npy(text_dir / "imagebind_options.npy", ib_options.astype(np.float32))
    del model
    torch.cuda.empty_cache()

    if args.dataset == "avqa":
        if not args.clip_source.is_dir():
            raise FileNotFoundError(args.clip_source)
        sys.path.insert(0, str(args.clip_source.resolve()))
        import clip_net.clip as clip

        clip_model, _ = clip.load(
            "ViT-L/14@336px",
            device=args.device,
            download_root=str(args.clip_cache),
        )
        clip_model.eval()

        def clip_encode(texts: list[str]) -> np.ndarray:
            tokens = clip.tokenize(texts, truncate=True).to(args.device)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.float16
            ):
                result = clip_model.encode_text(tokens)
            return result.detach().float().cpu().numpy()

        clip_question = _batched_text(questions, clip_encode, 128)
        clip_options = _batched_text(
            [value for group in options for value in group],
            clip_encode,
            128,
        ).reshape(len(rows), 4, -1)
        _atomic_npy(text_dir / "clip_question.npy", clip_question)
        _atomic_npy(text_dir / "clip_options.npy", clip_options)
    _write_json(text_dir / "qa_ids.json", qa_ids)


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *unused: Any) -> None:
        return None


def _extract_audio(video: str, output: Path, sample_rate: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            video,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
        stderr=subprocess.PIPE,
        text=True,
    )


def _imagebind_video_proxy(video: str, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            video,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "fps=2,scale=if(gt(iw\\,ih)\\,-2\\,256):"
            "if(gt(iw\\,ih)\\,256\\,-2)",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        stderr=subprocess.PIPE,
        text=True,
    )


def _valid_chunk(
    path: Path,
    ids: list[str],
    fields: dict[str, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            if values["ids"].astype(str).tolist() != ids:
                return False
            return all(
                values[name].shape == (len(ids), dim)
                and values[name].dtype == np.float16
                and np.isfinite(values[name]).all()
                for name, dim in fields.items()
            )
    except Exception:
        return False


def _assemble_chunks(
    chunks: list[list[dict[str, str]]],
    chunk_dir: Path,
    output_dir: Path,
    fields: dict[str, int],
) -> None:
    collected = {name: [] for name in fields}
    ids: list[str] = []
    for number, group in enumerate(chunks):
        group_ids = [_media_id(row) for row in group]
        path = chunk_dir / f"chunk_{number:05d}.npz"
        if not _valid_chunk(path, group_ids, fields):
            raise RuntimeError(f"missing or invalid feature chunk {number}")
        with np.load(path, allow_pickle=False) as values:
            ids.extend(values["ids"].astype(str).tolist())
            for name in fields:
                collected[name].append(values[name])
    for name, parts in collected.items():
        _atomic_npy(output_dir / f"{name}.npy", np.concatenate(parts))
    _write_json(output_dir / "clip_ids.json", ids)


def extract_imagebind_media(args: argparse.Namespace) -> None:
    torch = _require_cuda(args.device)
    rows = _load_media_rows(args.protocol_root)
    model, data, modality = _load_imagebind(
        args.imagebind_source,
        args.imagebind_checkpoint,
        args.device,
    )
    fields = {"vision": IMAGEBIND_DIM, "audio": IMAGEBIND_DIM}
    if args.dataset == "music_avqa":
        fields["fused"] = IMAGEBIND_DIM
    output = args.protocol_root / "features/imagebind"
    chunk_dir = output / "chunks"
    chunks = [
        rows[start : start + args.chunk_size]
        for start in range(0, len(rows), args.chunk_size)
    ]
    completed = 0
    for number, group in enumerate(chunks):
        if args.limit_chunks and number >= args.limit_chunks:
            break
        ids = [_media_id(row) for row in group]
        target = chunk_dir / f"chunk_{number:05d}.npz"
        if _valid_chunk(target, ids, fields):
            completed += len(group)
            continue
        vision_values: list[np.ndarray] = []
        audio_values: list[np.ndarray] = []
        fused_values: list[np.ndarray] = []
        infer_batch = args.imagebind_batch if args.dataset == "avqa" else 1
        for start in range(0, len(group), infer_batch):
            prepared = []
            for row in group[start : start + infer_batch]:
                work = Path(tempfile.mkdtemp(prefix=f"{_media_id(row)}_"))
                try:
                    wav = work / "audio.wav"
                    _extract_audio(row["media_path"], wav, 16_000)
                    try:
                        video = data.load_and_transform_video_data(
                            [row["media_path"]], args.device
                        )
                    except Exception:
                        if args.dataset != "music_avqa":
                            raise
                        proxy = work / "video_proxy.mp4"
                        _imagebind_video_proxy(row["media_path"], proxy)
                        video = data.load_and_transform_video_data(
                            [str(proxy)], args.device
                        )
                    audio = data.load_and_transform_audio_data([str(wav)], args.device)
                    prepared.append((work, video, audio))
                except Exception:
                    shutil.rmtree(work, ignore_errors=True)
                    raise
            try:
                video_batch = torch.cat([item[1] for item in prepared], dim=0)
                audio_batch = torch.cat([item[2] for item in prepared], dim=0)
                context = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if args.dataset == "avqa"
                    else _null_context()
                )
                with torch.inference_mode(), context:
                    result = model(
                        {
                            modality.VISION: video_batch,
                            modality.AUDIO: audio_batch,
                        }
                    )
                visual = result[modality.VISION].detach().float().cpu().numpy()
                audio = result[modality.AUDIO].detach().float().cpu().numpy()
                if args.dataset == "music_avqa":
                    visual = _l2(visual)
                    audio = _l2(audio)
                    fused = _l2((visual + audio) / 2.0)
                    fused_values.extend(fused.astype(np.float16))
                vision_values.extend(visual.astype(np.float16))
                audio_values.extend(audio.astype(np.float16))
            finally:
                for work, _, _ in prepared:
                    shutil.rmtree(work, ignore_errors=True)
                torch.cuda.empty_cache()
        values: dict[str, Any] = {
            "ids": np.asarray(ids),
            "vision": np.stack(vision_values),
            "audio": np.stack(audio_values),
        }
        if args.dataset == "music_avqa":
            values["fused"] = np.stack(fused_values)
        _atomic_npz(target, **values)
        completed += len(group)
        print(
            json.dumps(
                {
                    "stage": "imagebind_media",
                    "complete": completed,
                    "total": len(rows),
                    "chunk": number,
                }
            ),
            flush=True,
        )
    if not args.limit_chunks:
        _assemble_chunks(chunks, chunk_dir, output, fields)


def _video_frames(path: str, processor: Any) -> Any:
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                path,
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    )
    if not probe.get("streams"):
        raise RuntimeError(f"video stream not found: {path}")
    stream = probe["streams"][0]
    duration = max(float(probe.get("format", {}).get("duration") or 0.0), 0.1)
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            path,
            "-an",
            "-vf",
            f"fps={NUM_FRAMES}/{duration:.8f}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        stderr=subprocess.STDOUT,
    )
    width, height = int(stream["width"]), int(stream["height"])
    frame_size = width * height * 3
    count = len(raw) // frame_size
    if count < 1:
        raise RuntimeError(f"no decodable visual frame: {path}")
    frames = np.frombuffer(raw[: count * frame_size], np.uint8).reshape(
        count, height, width, 3
    )
    if count < NUM_FRAMES:
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], NUM_FRAMES - count, axis=0)]
        )
    elif count > NUM_FRAMES:
        indices = np.linspace(0, count - 1, NUM_FRAMES).round().astype(int)
        frames = frames[indices]
    return processor(
        [frame.copy() for frame in frames[:NUM_FRAMES]],
        return_tensors="pt",
    )["pixel_values"]


def _muq_waveform(path: str, work: Path) -> Any:
    import torch
    import torchaudio

    wav = work / "audio.wav"
    _extract_audio(path, wav, MUQ_SAMPLE_RATE)
    waveform, sample_rate = torchaudio.load(str(wav))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != MUQ_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, MUQ_SAMPLE_RATE
        )
    waveform = torch.nn.functional.pad(
        waveform, (0, max(0, MUQ_SAMPLES - waveform.shape[1]))
    )
    return waveform[:, :MUQ_SAMPLES]


def extract_muq_videomae_media(args: argparse.Namespace) -> None:
    torch = _require_cuda(args.device)
    from muq import MuQ
    from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor

    rows = _load_media_rows(args.protocol_root)
    processor = VideoMAEImageProcessor.from_pretrained(
        str(args.videomae_model), local_files_only=True
    )
    video_model = VideoMAEForPreTraining.from_pretrained(
        str(args.videomae_model),
        local_files_only=True,
        use_safetensors=True,
    ).to(args.device).eval()
    audio_model = MuQ.from_pretrained(str(args.muq_model)).to(args.device).eval()
    output = args.protocol_root / "features/muq_videomae"
    chunk_dir = output / "chunks"
    fields = {"muq": MUQ_DIM, "videomae": VIDEOMAE_DIM}
    chunks = [
        rows[start : start + args.chunk_size]
        for start in range(0, len(rows), args.chunk_size)
    ]
    completed = 0
    for number, group in enumerate(chunks):
        if args.limit_chunks and number >= args.limit_chunks:
            break
        ids = [_media_id(row) for row in group]
        target = chunk_dir / f"chunk_{number:05d}.npz"
        if _valid_chunk(target, ids, fields):
            completed += len(group)
            continue
        muq_values: list[np.ndarray] = []
        videomae_values: list[np.ndarray] = []
        infer_batch = args.media_batch if args.dataset == "avqa" else 1
        for start in range(0, len(group), infer_batch):
            prepared = []
            for row in group[start : start + infer_batch]:
                work = Path(tempfile.mkdtemp(prefix=f"{_media_id(row)}_"))
                try:
                    prepared.append(
                        (
                            work,
                            _video_frames(row["media_path"], processor),
                            _muq_waveform(row["media_path"], work),
                        )
                    )
                except Exception:
                    shutil.rmtree(work, ignore_errors=True)
                    raise
            try:
                pixels = torch.cat([item[1] for item in prepared], dim=0).to(
                    args.device
                )
                waveform = torch.cat([item[2] for item in prepared], dim=0).to(
                    args.device
                )
                context = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if args.dataset == "avqa"
                    else _null_context()
                )
                with torch.inference_mode(), context:
                    video = video_model.videomae(
                        pixel_values=pixels, return_dict=True
                    ).last_hidden_state.mean(1)
                    audio = audio_model(
                        waveform, output_hidden_states=True
                    ).last_hidden_state.mean(1)
                if args.dataset == "avqa" and not torch.isfinite(audio).all():
                    with torch.inference_mode():
                        audio = audio_model(
                            waveform, output_hidden_states=True
                        ).last_hidden_state.mean(1)
                if not torch.isfinite(audio).all():
                    raise RuntimeError("MuQ returned non-finite embeddings")
                video_array = video.detach().float().cpu().numpy()
                audio_array = audio.detach().float().cpu().numpy()
                if args.dataset == "music_avqa":
                    video_array = _l2(video_array)
                    audio_array = _l2(audio_array)
                videomae_values.extend(video_array.astype(np.float16))
                muq_values.extend(audio_array.astype(np.float16))
            finally:
                for work, _, _ in prepared:
                    shutil.rmtree(work, ignore_errors=True)
                torch.cuda.empty_cache()
        _atomic_npz(
            target,
            ids=np.asarray(ids),
            muq=np.stack(muq_values),
            videomae=np.stack(videomae_values),
        )
        completed += len(group)
        print(
            json.dumps(
                {
                    "stage": "muq_videomae_media",
                    "complete": completed,
                    "total": len(rows),
                    "chunk": number,
                }
            ),
            flush=True,
        )
    if not args.limit_chunks:
        _assemble_chunks(chunks, chunk_dir, output, fields)


def _music_split(row: dict[str, Any]) -> str:
    value = str(
        _first(row, ("experiment_split", "official_split", "split", "partition"))
    ).lower()
    aliases = {
        "adapt": "train",
        "validation": "val",
        "valid": "val",
        "tune": "val",
        "holdout": "test",
    }
    value = aliases.get(value, value)
    if value not in {"train", "val", "test"}:
        raise ValueError(
            f"MUSIC QA {_qa_id(row)} has unsupported split {value!r}; "
            "expected train, val, or test"
        )
    return value


def _music_answer(row: dict[str, Any]) -> str:
    return str(_first(row, ("gold_answer", "answer", "anser", "label")))


def pack_music(args: argparse.Namespace) -> None:
    root = args.protocol_root
    rows = _load_qa_rows(root)
    qa_ids = json.loads((root / "features/text/qa_ids.json").read_text())
    questions = np.load(root / "features/text/imagebind_question.npy")
    if [_qa_id(row) for row in rows] != [str(value) for value in qa_ids]:
        raise ValueError("MUSIC text features are not aligned with the QA manifest")
    train_answers = {
        _music_answer(row) for row in rows if _music_split(row) == "train"
    }
    vocab = np.asarray(sorted(train_answers))
    if len(vocab) != 42:
        raise ValueError(
            f"MUSIC training partition must define exactly 42 answers, got {len(vocab)}"
        )
    answer_index = {answer: index for index, answer in enumerate(vocab.tolist())}
    answers = [_music_answer(row) for row in rows]
    unknown = sorted(set(answers) - set(answer_index))
    if unknown:
        raise ValueError(f"MUSIC non-training rows contain unknown answers: {unknown}")
    labels = np.asarray([answer_index[value] for value in answers], dtype=np.int64)
    split = np.asarray([_music_split(row) for row in rows])
    indices = {
        "train_idx": np.flatnonzero(split == "train"),
        "val_idx": np.flatnonzero(split == "val"),
        "test_idx": np.flatnonzero(split == "test"),
    }
    media_rows = _load_media_rows(root)
    media_ids = [_media_id(row) for row in media_rows]
    media_index = {value: index for index, value in enumerate(media_ids)}
    if len(media_index) != len(media_ids):
        raise ValueError("full_media_manifest.csv contains duplicate media IDs")
    qa_media_indices = np.asarray(
        [media_index[_media_id(row)] for row in rows], dtype=np.int64
    )
    output = root / "features/music_arrays"
    expert_media = {
        "imagebind": np.load(root / "features/imagebind/fused.npy", mmap_mode="r"),
        "muq": np.load(root / "features/muq_videomae/muq.npy", mmap_mode="r"),
        "videomae": np.load(
            root / "features/muq_videomae/videomae.npy", mmap_mode="r"
        ),
    }
    for expert, media in expert_media.items():
        if len(media) != len(media_ids):
            raise ValueError(f"{expert} media features do not match the media manifest")
        _atomic_npz(
            output / f"{expert}_arrays.npz",
            media=np.asarray(media[qa_media_indices], dtype=np.float32),
            question=np.asarray(questions, dtype=np.float32),
            y=labels,
            vocab=vocab,
            **indices,
        )
    manifest_rows = [
        {
            "qa_id": _qa_id(row),
            "source_id": str(_first(row, ("source_id", "video_id", "clip_id"))),
            "media_id": _media_id(row),
            "split": _music_split(row),
        }
        for row in rows
    ]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "music_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "vocab": len(vocab),
                "output": str(output),
            }
        )
    )


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("avqa", "music_avqa"), required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-protocol frozen expert feature extraction"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    text = commands.add_parser("text")
    _common_parser(text)
    text.add_argument("--imagebind-source", type=Path, required=True)
    text.add_argument("--imagebind-checkpoint", type=Path, required=True)
    text.add_argument("--clip-source", type=Path, default=Path("."))
    text.add_argument(
        "--clip-cache",
        type=Path,
        default=Path("model/experts/CLIP/cache"),
    )
    text.add_argument("--text-batch-size", type=int, default=256)
    text.set_defaults(handler=extract_text)

    imagebind = commands.add_parser("imagebind")
    _common_parser(imagebind)
    imagebind.add_argument("--imagebind-source", type=Path, required=True)
    imagebind.add_argument("--imagebind-checkpoint", type=Path, required=True)
    imagebind.add_argument("--chunk-size", type=int, default=32)
    imagebind.add_argument("--imagebind-batch", type=int, choices=(1, 2), default=2)
    imagebind.add_argument("--limit-chunks", type=int, default=0)
    imagebind.set_defaults(handler=extract_imagebind_media)

    other = commands.add_parser("muq-videomae")
    _common_parser(other)
    other.add_argument("--muq-model", type=Path, required=True)
    other.add_argument("--videomae-model", type=Path, required=True)
    other.add_argument("--chunk-size", type=int, default=32)
    other.add_argument("--media-batch", type=int, choices=(1, 2, 4), default=4)
    other.add_argument("--limit-chunks", type=int, default=0)
    other.set_defaults(handler=extract_muq_videomae_media)

    pack = commands.add_parser("pack-music")
    pack.add_argument("--protocol-root", type=Path, required=True)
    pack.set_defaults(handler=pack_music)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    args.handler(args)
    print(
        json.dumps(
            {
                "command": args.command,
                "status": "complete",
                "elapsed_seconds": round(time.time() - started, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
