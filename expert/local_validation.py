#!/usr/bin/env python3
"""Run paper-style local validation with the trained MUSIC expert heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from expert.extract_raw_features import (  # noqa: E402
    _extract_audio,
    _imagebind_video_proxy,
    _load_imagebind,
    _muq_waveform,
    _video_frames,
)
from expert.music_avqa_42 import MusicAVQA42Head  # noqa: E402
from sde_cctdh.artifacts import write_jsonl  # noqa: E402
from sde_cctdh.intervention import (  # noqa: E402
    intervention_effect,
    intervention_operator,
    low_saliency_control_window,
    saliency_l2,
    top_window,
)
from sde_cctdh.runtime import read_table  # noqa: E402


SUPPORTED_EXPERTS = ("imagebind", "muq", "videomae")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _cache_path(root: Path, expert: str, source_id: str) -> Path:
    digest = hashlib.sha256(source_id.encode()).hexdigest()[:24]
    return root / expert / f"{digest}.pt"


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_music_arrays(
    protocol_root: Path,
    expert: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    arrays_path = (
        protocol_root / "features/music_arrays" / f"{expert}_arrays.npz"
    )
    manifest_path = (
        protocol_root / "features/music_arrays/music_manifest.jsonl"
    )
    with np.load(arrays_path, allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}
    manifest = read_table(manifest_path).reset_index(drop=True)
    if len(manifest) != len(arrays["y"]):
        raise ValueError("MUSIC arrays and manifest are not row-aligned")
    return arrays, manifest


def _oof_fold_map(head_dir: Path, expert: str) -> dict[int, int]:
    result: dict[int, int] = {}
    checkpoints = sorted(head_dir.glob(f"{expert}_fold*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no {expert} OOF checkpoints under {head_dir}")
    for checkpoint in checkpoints:
        fold = int(checkpoint.stem.rsplit("fold", 1)[1])
        heldout = head_dir / f"{expert}_fold{fold}_heldout.npz"
        with np.load(heldout, allow_pickle=False) as payload:
            for row_index in payload["row_indices"].astype(int):
                if int(row_index) in result:
                    raise ValueError("an expert row appears in multiple OOF folds")
                result[int(row_index)] = fold
    return result


def _load_head(
    head_dir: Path,
    expert: str,
    fold: int,
    *,
    media_dim: int,
    question_dim: int,
    device: torch.device,
) -> tuple[MusicAVQA42Head, dict[str, np.ndarray], list[str]]:
    payload = torch.load(
        head_dir / f"{expert}_fold{fold}.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    model = MusicAVQA42Head(
        media_dim,
        question_dim,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["model_state"])
    model.requires_grad_(False).eval().to(device)
    normalizer = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalizer"].items()
    }
    return model, normalizer, list(map(str, payload["vocab"]))


class LocalEncoder:
    def __init__(
        self,
        *,
        expert: str,
        model_path: Path,
        cache_root: Path,
        device: torch.device,
    ) -> None:
        self.expert = expert
        self.cache_root = cache_root
        self.device = device
        self.model_path = model_path.expanduser().resolve()
        if expert == "muq":
            from muq import MuQ

            self.processor = None
            self.model = MuQ.from_pretrained(str(self.model_path)).to(device).eval()
        elif expert == "videomae":
            from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor

            self.processor = VideoMAEImageProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self.model = VideoMAEForPreTraining.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                use_safetensors=True,
            ).to(device).eval()
        else:
            raise ValueError(f"unsupported local encoder: {expert}")
        self.model.requires_grad_(False)

    def encode(self, source_id: str, media_path: str) -> torch.Tensor:
        cache = _cache_path(self.cache_root, self.expert, source_id)
        media = Path(media_path).expanduser().resolve()
        media_stat = media.stat()
        if cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            if (
                payload.get("source_id") == source_id
                and payload.get("media_path") == str(media)
                and payload.get("media_bytes") == media_stat.st_size
                and payload.get("media_mtime_ns") == media_stat.st_mtime_ns
                and payload.get("model_path") == str(self.model_path)
            ):
                return payload["local_hidden"].float()
        work = Path(tempfile.mkdtemp(prefix=f"{self.expert}_local_"))
        try:
            if self.expert == "muq":
                values = _muq_waveform(str(media), work).to(self.device)
                with torch.inference_mode():
                    local = self.model(
                        values,
                        output_hidden_states=True,
                    ).last_hidden_state[0]
            else:
                values = _video_frames(str(media), self.processor).to(self.device)
                with torch.inference_mode():
                    local = self.model.videomae(
                        pixel_values=values,
                        return_dict=True,
                    ).last_hidden_state[0]
            result = local.detach().float().cpu()
            _atomic_torch(
                cache,
                {
                    "expert": self.expert,
                    "source_id": source_id,
                    "media_path": str(media),
                    "media_bytes": media_stat.st_size,
                    "media_mtime_ns": media_stat.st_mtime_ns,
                    "model_path": str(self.model_path),
                    "local_hidden": result.to(torch.float16),
                    "pooling": "last_hidden_state.mean(dim=0)",
                },
            )
            return result
        finally:
            shutil.rmtree(work, ignore_errors=True)


class ImageBindLocalEncoder:
    """Replay ImageBind from its local vision/audio block-11 activations."""

    hook_layer = 11

    def __init__(
        self,
        *,
        source_path: Path,
        checkpoint_path: Path,
        cache_root: Path,
        device: torch.device,
    ) -> None:
        self.source_path = source_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.cache_root = cache_root
        self.device = device
        self.model, self.data, self.modality = _load_imagebind(
            self.source_path,
            self.checkpoint_path,
            str(device),
        )
        self.model.requires_grad_(False).eval()

    @staticmethod
    def _reduce_shape(value: torch.Tensor) -> tuple[int, int]:
        return tuple(map(int, value.shape[:2])) if value.ndim >= 5 else (int(value.shape[0]), 1)

    def _capture(self, media_path: Path) -> dict[str, Any]:
        work = Path(tempfile.mkdtemp(prefix="imagebind_local_"))
        hooks = []
        captured: dict[str, torch.Tensor] = {}
        try:
            wav = work / "audio.wav"
            _extract_audio(str(media_path), wav, 16_000)
            try:
                video = self.data.load_and_transform_video_data(
                    [str(media_path)], str(self.device)
                )
            except Exception:
                proxy = work / "video_proxy.mp4"
                _imagebind_video_proxy(str(media_path), proxy)
                video = self.data.load_and_transform_video_data(
                    [str(proxy)], str(self.device)
                )
            audio = self.data.load_and_transform_audio_data(
                [str(wav)], str(self.device)
            )

            def save(name: str):
                def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
                    captured[name] = output.detach().float().cpu()

                return hook

            hooks.append(
                self.model.modality_trunks[self.modality.VISION]
                .blocks[self.hook_layer]
                .register_forward_hook(save("vision_hidden"))
            )
            hooks.append(
                self.model.modality_trunks[self.modality.AUDIO]
                .blocks[self.hook_layer]
                .register_forward_hook(save("audio_hidden"))
            )
            with torch.inference_mode():
                self.model(
                    {
                        self.modality.VISION: video,
                        self.modality.AUDIO: audio,
                    }
                )
            if set(captured) != {"vision_hidden", "audio_hidden"}:
                raise RuntimeError("ImageBind local hooks did not capture both modalities")
            return {
                "vision_hidden": captured["vision_hidden"].to(torch.float16),
                "audio_hidden": captured["audio_hidden"].to(torch.float16),
                "vision_reduce_shape": self._reduce_shape(video),
                "audio_reduce_shape": self._reduce_shape(audio),
            }
        finally:
            for handle in hooks:
                handle.remove()
            shutil.rmtree(work, ignore_errors=True)

    def encode(self, source_id: str, media_path: str) -> dict[str, Any]:
        cache = _cache_path(self.cache_root, "imagebind", source_id)
        media = Path(media_path).expanduser().resolve()
        media_stat = media.stat()
        checkpoint_stat = self.checkpoint_path.stat()
        if cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            if (
                payload.get("source_id") == source_id
                and payload.get("media_path") == str(media)
                and payload.get("media_bytes") == media_stat.st_size
                and payload.get("media_mtime_ns") == media_stat.st_mtime_ns
                and payload.get("checkpoint_path") == str(self.checkpoint_path)
                and payload.get("checkpoint_bytes") == checkpoint_stat.st_size
                and payload.get("hook_layer") == self.hook_layer
            ):
                return payload
        payload = self._capture(media)
        payload.update(
            {
                "expert": "imagebind",
                "source_id": source_id,
                "media_path": str(media),
                "media_bytes": media_stat.st_size,
                "media_mtime_ns": media_stat.st_mtime_ns,
                "source_path": str(self.source_path),
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_bytes": checkpoint_stat.st_size,
                "hook_layer": self.hook_layer,
            }
        )
        _atomic_torch(cache, payload)
        return payload

    def finish_modality(
        self,
        hidden: torch.Tensor,
        modality: str,
        reduce_shape: tuple[int, int],
    ) -> torch.Tensor:
        trunk = self.model.modality_trunks[modality]
        value = hidden
        for block in trunk.blocks[self.hook_layer + 1 :]:
            value = block(value)
        if trunk.post_transformer_layer is not None:
            value = trunk.post_transformer_layer(value)
        value = self.model.modality_heads[modality](value)
        value = self.model.modality_postprocessors[modality](value)
        batch, segments = reduce_shape
        if segments > 1:
            value = value.reshape(batch, segments, -1).mean(dim=1)
        return F.normalize(value.float(), dim=-1, eps=1e-12)

    def fused_feature(
        self,
        vision_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        payload: dict[str, Any],
    ) -> torch.Tensor:
        vision = self.finish_modality(
            vision_hidden,
            self.modality.VISION,
            tuple(payload["vision_reduce_shape"]),
        )
        audio = self.finish_modality(
            audio_hidden,
            self.modality.AUDIO,
            tuple(payload["audio_reduce_shape"]),
        )
        return F.normalize((vision + audio) / 2.0, dim=-1, eps=1e-12)[0]


def _replacement_window(
    donor: torch.Tensor,
    indices: torch.Tensor,
    length: int,
) -> torch.Tensor:
    selected = donor[indices]
    if selected.shape[0] == length:
        return selected
    positions = torch.linspace(
        0,
        max(0, selected.shape[0] - 1),
        length,
        device=selected.device,
    ).round().long()
    return selected[positions]


def _common_bins(
    saliency: dict[str, torch.Tensor],
    window_fraction: float,
) -> tuple[int, int, dict[str, list[torch.Tensor]]]:
    if not 0 < window_fraction <= 0.5:
        raise ValueError("window_fraction must be in (0, 0.5]")
    bin_count = max(2, int(round(1.0 / window_fraction)))
    combined = torch.zeros(bin_count, device=next(iter(saliency.values())).device)
    windows: dict[str, list[torch.Tensor]] = {}
    for modality, values in saliency.items():
        if values.numel() < 2:
            raise ValueError(f"{modality} local sequence is too short")
        modality_windows = []
        scores = []
        for index in range(bin_count):
            start = int(round(index * values.numel() / bin_count))
            stop = int(round((index + 1) * values.numel() / bin_count))
            stop = max(start + 1, min(stop, values.numel()))
            indices = torch.arange(start, stop, device=values.device)
            modality_windows.append(indices)
            scores.append(values[indices].mean())
        score_tensor = torch.stack(scores)
        combined += score_tensor / score_tensor.norm().clamp_min(1e-12)
        windows[modality] = modality_windows
    return int(combined.argmax()), int(combined.argmin()), windows


def _imagebind_replacement(
    donor: torch.Tensor,
    donor_indices: torch.Tensor,
    target_selected: torch.Tensor,
) -> torch.Tensor:
    selected = _replacement_window(donor, donor_indices, target_selected.shape[0])
    if selected.shape[1] != target_selected.shape[1]:
        selected = selected.mean(dim=1, keepdim=True).expand(
            -1, target_selected.shape[1], -1
        )
    return selected


def _imagebind_artifacts(
    *,
    encoder: ImageBindLocalEncoder,
    current: dict[str, Any],
    donor: dict[str, Any],
    head: MusicAVQA42Head,
    normalizer: dict[str, np.ndarray],
    question: torch.Tensor,
    gold_index: int,
    window_fraction: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    media_mean = torch.as_tensor(normalizer["media_mean"], device=device)
    media_std = torch.as_tensor(normalizer["media_std"], device=device)

    def tensors(payload: dict[str, Any], *, gradients: bool) -> dict[str, torch.Tensor]:
        result = {
            name: payload[f"{name}_hidden"].float().to(device)
            for name in ("vision", "audio")
        }
        for value in result.values():
            value.requires_grad_(gradients)
        return result

    def logits(values: dict[str, torch.Tensor], payload: dict[str, Any]) -> torch.Tensor:
        pooled = encoder.fused_feature(values["vision"], values["audio"], payload)
        standardized = (pooled - media_mean) / media_std.clamp_min(1e-6)
        return head(standardized.unsqueeze(0), question)[0]

    def gold_score(values: dict[str, torch.Tensor], payload: dict[str, Any]) -> torch.Tensor:
        return torch.log_softmax(logits(values, payload).float(), dim=-1)[gold_index]

    local = tensors(current, gradients=True)
    original = gold_score(local, current)
    if int(logits(local, current).argmax()) != gold_index:
        raise RuntimeError(
            "ImageBind local encoder/head replay differs from the OOF support audit"
        )
    gradients = torch.autograd.grad(
        original,
        (local["vision"], local["audio"]),
        retain_graph=True,
    )
    saliency = {
        name: gradient.norm(dim=-1).mean(dim=1)
        for name, gradient in zip(("vision", "audio"), gradients)
    }
    target_bin, control_bin, windows = _common_bins(saliency, window_fraction)

    donor_values = tensors(donor, gradients=True)
    donor_original = gold_score(donor_values, donor)
    donor_gradients = torch.autograd.grad(
        donor_original,
        (donor_values["vision"], donor_values["audio"]),
    )
    donor_saliency = {
        name: gradient.norm(dim=-1).mean(dim=1)
        for name, gradient in zip(("vision", "audio"), donor_gradients)
    }
    donor_bin, _, donor_windows = _common_bins(donor_saliency, window_fraction)

    values: dict[str, float] = {}
    selected_bins = {
        "target": target_bin,
        "control": control_bin,
        "donor": donor_bin,
    }
    for role, bin_index in selected_bins.items():
        for operator in ("mask", "replace"):
            changed = {name: value.detach().clone() for name, value in local.items()}
            for name in ("vision", "audio"):
                indices = windows[name][bin_index]
                if operator == "mask":
                    changed[name][indices] = 0.0
                else:
                    changed[name][indices] = _imagebind_replacement(
                        donor_values[name].detach(),
                        donor_windows[name][donor_bin],
                        changed[name][indices],
                    )
            values[f"{role}_{operator}"] = intervention_effect(
                original,
                gold_score(changed, current),
            )

    detail = {}
    for role, bin_index in (("target", target_bin), ("control", control_bin)):
        pooled = [
            local[name].detach()[windows[name][bin_index]]
            .mean(dim=(0, 1))
            .float()
            .cpu()
            for name in ("vision", "audio")
        ]
        detail[f"{role}_pooled_feature"] = torch.cat(pooled).tolist()
    metadata = values | {
        "target_bin": target_bin,
        "control_bin": control_bin,
        "donor_bin": donor_bin,
        "target_indices": {
            name: windows[name][target_bin].cpu().tolist()
            for name in ("vision", "audio")
        },
        "control_indices": {
            name: windows[name][control_bin].cpu().tolist()
            for name in ("vision", "audio")
        },
        "donor_indices": {
            name: donor_windows[name][donor_bin].cpu().tolist()
            for name in ("vision", "audio")
        },
        "window_fraction": window_fraction,
        "hook_layer": encoder.hook_layer,
        "saliency": "gold_log_probability_gradient_l2",
        "detail_layout": "vision_block11_then_audio_block11",
    }
    return metadata, detail


def validate(args: argparse.Namespace) -> dict[str, Any]:
    _seed(args.seed)
    device = torch.device(args.device)
    arrays, manifest = _load_music_arrays(args.protocol_root, args.expert)
    manifest_index = {
        str(qa_id): index
        for index, qa_id in enumerate(manifest["qa_id"].astype(str))
    }
    fold_map = _oof_fold_map(args.head_dir, args.expert)
    pairs = read_table(args.pair_audit)
    required = {
        "qa_id",
        "source_id",
        "media_path",
        "expert_name",
        "expert_support",
    }
    if required - set(pairs.columns):
        raise ValueError(f"pair audit lacks fields: {sorted(required - set(pairs.columns))}")
    rows = pairs[
        pairs["expert_name"].astype(str).eq(args.expert)
        & pairs["expert_support"].astype(bool)
    ].copy()
    if rows.empty:
        raise ValueError(f"no supported {args.expert} pairs to validate")
    rows = rows.sort_values(["source_id", "qa_id"], kind="stable")
    source_media = (
        rows.groupby("source_id", sort=True)["media_path"].first().astype(str).to_dict()
    )
    sources = sorted(source_media)
    if len(sources) < 2:
        raise ValueError("local validation requires at least two source IDs")
    ordered = sorted(
        sources,
        key=lambda value: hashlib.sha256(
            f"{args.seed}|{value}".encode()
        ).hexdigest(),
    )
    donors = {
        source: ordered[(ordered.index(source) + 1) % len(ordered)]
        for source in ordered
    }
    if args.expert == "imagebind":
        if args.imagebind_source is None or args.imagebind_checkpoint is None:
            raise ValueError(
                "ImageBind validation requires --imagebind-source and "
                "--imagebind-checkpoint"
            )
        encoder: LocalEncoder | ImageBindLocalEncoder = ImageBindLocalEncoder(
            source_path=args.imagebind_source,
            checkpoint_path=args.imagebind_checkpoint,
            cache_root=args.cache,
            device=device,
        )
    else:
        if args.model_path is None:
            raise ValueError(f"{args.expert} validation requires --model-path")
        encoder = LocalEncoder(
            expert=args.expert,
            model_path=args.model_path,
            cache_root=args.cache,
            device=device,
        )
    heads: dict[int, tuple[MusicAVQA42Head, dict[str, np.ndarray], list[str]]] = {}
    effects = []
    details = []
    for ordinal, row in enumerate(rows.to_dict("records"), 1):
        qa_id = str(row["qa_id"])
        row_index = manifest_index.get(qa_id)
        if row_index is None or row_index not in fold_map:
            raise ValueError(f"{qa_id}: no source-held-out OOF expert fold")
        fold = fold_map[row_index]
        if fold not in heads:
            heads[fold] = _load_head(
                args.head_dir,
                args.expert,
                fold,
                media_dim=int(arrays["media"].shape[1]),
                question_dim=int(arrays["question"].shape[1]),
                device=device,
            )
        head, normalizer, vocab = heads[fold]
        gold_answer = str(row["gold_answer"])
        if gold_answer not in vocab:
            raise ValueError(f"{qa_id}: gold answer is absent from expert vocabulary")
        gold_index = vocab.index(gold_answer)
        question = torch.as_tensor(
            (
                arrays["question"][row_index].astype(np.float32)
                - normalizer["question_mean"]
            )
            / normalizer["question_std"],
            device=device,
        ).unsqueeze(0)
        media_mean = torch.as_tensor(
            normalizer["media_mean"],
            device=device,
        )
        media_std = torch.as_tensor(
            normalizer["media_std"],
            device=device,
        )

        def expert_logits(local_value: torch.Tensor) -> torch.Tensor:
            pooled = torch.nn.functional.normalize(
                local_value.mean(dim=0),
                dim=0,
                eps=1e-12,
            )
            standardized = (pooled - media_mean) / media_std.clamp_min(1e-6)
            return head(standardized.unsqueeze(0), question)[0]

        def gold_log_probability(local_value: torch.Tensor) -> torch.Tensor:
            return torch.log_softmax(
                expert_logits(local_value).float(),
                dim=-1,
            )[gold_index]

        source = str(row["source_id"])
        donor_source = donors[source]
        common = {
            "qa_id": qa_id,
            "source_id": source,
            "expert_name": args.expert,
        }
        if args.expert == "imagebind":
            assert isinstance(encoder, ImageBindLocalEncoder)
            current_payload = encoder.encode(source, str(row["media_path"]))
            donor_payload = encoder.encode(
                donor_source,
                source_media[donor_source],
            )
            effect_metadata, detail_metadata = _imagebind_artifacts(
                encoder=encoder,
                current=current_payload,
                donor=donor_payload,
                head=head,
                normalizer=normalizer,
                question=question,
                gold_index=gold_index,
                window_fraction=args.window_fraction,
                device=device,
            )
            effects.append(
                common | effect_metadata | {"donor_source_id": donor_source}
            )
            details.append(common | detail_metadata)
            if ordinal % args.flush_every == 0 or ordinal == len(rows):
                write_jsonl(
                    args.output / "imagebind_intervention_effects.jsonl",
                    pd.DataFrame(effects),
                )
                write_jsonl(
                    args.output / "imagebind_local_details.jsonl",
                    pd.DataFrame(details),
                )
                print(
                    json.dumps(
                        {
                            "expert": args.expert,
                            "complete": ordinal,
                            "total": len(rows),
                        }
                    ),
                    flush=True,
                )
            continue

        assert isinstance(encoder, LocalEncoder)
        local = encoder.encode(source, str(row["media_path"])).to(device)
        local.requires_grad_(True)
        if int(expert_logits(local).argmax()) != gold_index:
            raise RuntimeError(
                f"{qa_id}: local encoder/head replay differs from the OOF support audit"
            )
        original = gold_log_probability(local)
        saliency = saliency_l2(original, local)
        window_length = max(1, int(round(local.shape[0] * args.window_fraction)))
        window_length = min(window_length, max(1, local.shape[0] // 2))
        target = top_window(saliency, window_length)
        control = low_saliency_control_window(
            saliency,
            target,
            window_length,
        )

        donor = encoder.encode(
            donor_source,
            source_media[donor_source],
        ).to(device)
        donor.requires_grad_(True)
        donor_saliency = saliency_l2(
            gold_log_probability(donor),
            donor,
        )
        donor_length = min(window_length, donor.shape[0])
        donor_window = top_window(donor_saliency, donor_length)
        replacement = _replacement_window(
            donor.detach(),
            donor_window,
            window_length,
        )
        normalized_start = int(
            round(
                int(donor_window[0])
                / max(1, donor.shape[0] - donor_length)
                * max(0, local.shape[0] - window_length)
            )
        )
        donor_control = torch.arange(
            normalized_start,
            normalized_start + window_length,
            device=device,
        )

        values: dict[str, float] = {}
        for name, indices in (
            ("target", target),
            ("control", control),
            ("donor", donor_control),
        ):
            for operator in ("mask", "replace"):
                changed = intervention_operator(
                    local.detach(),
                    indices,
                    operator=operator,
                    mask_value=0.0,
                    replacement=replacement if operator == "replace" else None,
                )
                values[f"{name}_{operator}"] = intervention_effect(
                    original,
                    gold_log_probability(changed),
                )
        effects.append(
            common
            | values
            | {
                "target_indices": target.detach().cpu().tolist(),
                "control_indices": control.detach().cpu().tolist(),
                "donor_source_id": donor_source,
                "donor_indices": donor_window.detach().cpu().tolist(),
                "window_length": window_length,
                "saliency": "gold_log_probability_gradient_l2",
            }
        )
        details.append(
            common
            | {
                "target_pooled_feature": local.detach()[target]
                .mean(dim=0)
                .float()
                .cpu()
                .tolist(),
                "control_pooled_feature": local.detach()[control]
                .mean(dim=0)
                .float()
                .cpu()
                .tolist(),
            }
        )
        if ordinal % args.flush_every == 0 or ordinal == len(rows):
            write_jsonl(
                args.output / f"{args.expert}_intervention_effects.jsonl",
                pd.DataFrame(effects),
            )
            write_jsonl(
                args.output / f"{args.expert}_local_details.jsonl",
                pd.DataFrame(details),
            )
            print(
                json.dumps(
                    {
                        "expert": args.expert,
                        "complete": ordinal,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    return {
        "stage": "local_validation",
        "expert": args.expert,
        "rows": len(rows),
        "effects": str(args.output / f"{args.expert}_intervention_effects.jsonl"),
        "details": str(args.output / f"{args.expert}_local_details.jsonl"),
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    effect_frames = []
    detail_frames = []
    for directory in args.input:
        effect_frames.extend(
            read_table(path)
            for path in sorted(directory.glob("*_intervention_effects.jsonl"))
        )
        detail_frames.extend(
            read_table(path)
            for path in sorted(directory.glob("*_local_details.jsonl"))
        )
    if not effect_frames or not detail_frames:
        raise ValueError("no local-validation artifacts found")
    effects = pd.concat(effect_frames, ignore_index=True).drop_duplicates(
        ["qa_id", "expert_name"],
        keep="last",
    )
    details = pd.concat(detail_frames, ignore_index=True).drop_duplicates(
        ["qa_id", "expert_name"],
        keep="last",
    )
    write_jsonl(args.output / "intervention_effects.jsonl", effects)
    write_jsonl(args.output / "local_details.jsonl", details)
    return {
        "stage": "merge_local_validation",
        "effect_rows": len(effects),
        "detail_rows": len(details),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("validate")
    run.add_argument("--expert", choices=SUPPORTED_EXPERTS, required=True)
    run.add_argument("--protocol-root", type=Path, required=True)
    run.add_argument("--head-dir", type=Path, required=True)
    run.add_argument("--pair-audit", type=Path, required=True)
    run.add_argument("--model-path", type=Path)
    run.add_argument("--imagebind-source", type=Path)
    run.add_argument("--imagebind-checkpoint", type=Path)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--window-fraction", type=float, default=0.125)
    run.add_argument("--flush-every", type=int, default=10)
    run.add_argument("--seed", type=int, default=18)
    run.add_argument("--device", default="cuda")
    run.set_defaults(function=validate)

    combine = commands.add_parser("merge")
    combine.add_argument("--input", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(function=merge)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(arguments.function(arguments), indent=2))
