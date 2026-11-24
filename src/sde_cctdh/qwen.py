"""Qwen2.5-Omni full-response runtime for the two-stage paper method."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .training import flat_autograd


IGNORE_INDEX = -100


def _torch_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_sha256(path: Path, name: str) -> str:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    generation = config.get("generation", {})
    material = {
        "protocol": config["protocols"][name],
        "generation": {
            "do_sample": generation.get("do_sample"),
            "temperature": generation.get("temperature"),
            "max_new_tokens": generation.get("max_new_tokens"),
            "qwen_video": generation.get("qwen_video"),
        },
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _adapter_file(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    return value / "adapter.pt" if value.is_dir() else value


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if value is None:
        return default
    return value


@dataclass(frozen=True)
class QwenProtocol:
    name: str
    dataset: str
    prompt: str
    answer_vocabulary: tuple[str, ...]
    fps: float = 1.0
    min_frames: int = 4
    max_frames: int = 128
    total_pixels: int = 12_845_056
    max_new_tokens: int = 8

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        name: str = "music_avqa_42",
    ) -> "QwenProtocol":
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        profile = config["protocols"][name]
        generation = config.get("generation", {})
        video = generation.get("qwen_video", {})
        vocabulary = profile.get("answer_vocabulary")
        if profile["dataset"] == "avqa":
            vocabulary = ["A", "B", "C", "D"]
        if not vocabulary:
            raise ValueError(f"protocol {name!r} has no answer vocabulary")
        return cls(
            name=name,
            dataset=str(profile["dataset"]),
            prompt=str(profile["prompt"]),
            answer_vocabulary=tuple(map(str, vocabulary)),
            fps=float(video.get("fps", 1.0)),
            min_frames=int(video.get("min_frames", 4)),
            max_frames=int(video.get("max_frames", 128)),
            total_pixels=int(video.get("total_pixels", 12_845_056)),
            max_new_tokens=int(
                generation.get("max_new_tokens", {}).get(
                    "avqa" if profile["dataset"] == "avqa" else "music_avqa",
                    8,
                )
            ),
        )

    def render_prompt(self, row: dict[str, Any]) -> str:
        question = str(_row_value(row, "question", _row_value(row, "question_text", "")))
        if self.dataset == "avqa":
            options = _row_value(row, "options", _row_value(row, "multi_choice"))
            if not isinstance(options, (list, tuple)) or len(options) != 4:
                raise ValueError(f"{row.get('qa_id')}: AVQA requires four options")
            return self.prompt.format(
                question=question,
                option_a=options[0],
                option_b=options[1],
                option_c=options[2],
                option_d=options[3],
            )
        answers = ", ".join(self.answer_vocabulary)
        return self.prompt.format(answers=answers, question=question)

    def normalize_answer(self, raw: Any) -> str:
        if raw is None:
            return "__invalid__"
        value = unicodedata.normalize("NFKC", str(raw)).strip()
        if self.dataset == "avqa":
            match = re.fullmatch(r"\s*([A-Da-d])[\s.)]*", value)
            return match.group(1).upper() if match else "__invalid__"
        value = re.sub(r"\s+", " ", value.casefold())
        value = "electric_bass" if value == "electric bass" else value
        return value if value in set(self.answer_vocabulary) else "__invalid__"

    def gold_response(self, row: dict[str, Any]) -> str:
        value = _row_value(row, "gold_response", _row_value(row, "gold_answer"))
        if value is None and self.dataset == "avqa":
            index = _row_value(row, "gold_index", _row_value(row, "answer"))
            if index is not None:
                value = "ABCD"[int(index)]
        normalized = self.normalize_answer(value)
        if normalized == "__invalid__":
            raise ValueError(f"{row.get('qa_id')}: invalid gold response {value!r}")
        return normalized


class LoRALinear(nn.Module):
    """Minimal mergeable LoRA wrapper used in both paper stages."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        update = F.linear(
            F.linear(self.dropout(value).to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return base + update.to(base.dtype) * self.scaling


class ResidualLoRALinear(nn.Module):
    """Residual LoRA stacked on a frozen SDE LoRA parent."""

    def __init__(
        self,
        parent: LoRALinear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.parent = parent
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        base = parent.base
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        parent = self.parent(value)
        update = F.linear(
            F.linear(self.dropout(value).to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return parent + update.to(parent.dtype) * self.scaling


def _target_modules(model: Any, layers: Iterable[int]) -> Iterable[tuple[str, Any, str]]:
    for layer_index in layers:
        attention = model.thinker.model.layers[int(layer_index)].self_attn
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield (
                f"thinker.model.layers.{int(layer_index)}.self_attn.{projection}",
                attention,
                projection,
            )


def install_sde_lora(
    model: Any,
    *,
    layers: Iterable[int],
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, LoRALinear]:
    modules: dict[str, LoRALinear] = {}
    for name, owner, attribute in _target_modules(model, layers):
        base = getattr(owner, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"unexpected Qwen LoRA target {name}: {type(base)!r}")
        wrapped = LoRALinear(
            base,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(owner, attribute, wrapped)
        modules[name] = wrapped
    return modules


def install_residual_lora(
    model: Any,
    *,
    layers: Iterable[int],
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, ResidualLoRALinear]:
    modules: dict[str, ResidualLoRALinear] = {}
    for name, owner, attribute in _target_modules(model, layers):
        parent = getattr(owner, attribute)
        if not isinstance(parent, LoRALinear):
            raise TypeError(f"SDE parent LoRA is missing at {name}")
        wrapped = ResidualLoRALinear(
            parent,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(owner, attribute, wrapped)
        modules[name] = wrapped
    return modules


def adapter_state(modules: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{suffix}": getattr(module, suffix).detach().cpu()
        for name, module in modules.items()
        for suffix in ("lora_A", "lora_B")
    }


def load_adapter_state(
    modules: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> None:
    expected = {
        f"{name}.{suffix}"
        for name in modules
        for suffix in ("lora_A", "lora_B")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f"adapter keys differ; missing={missing[:3]}, extra={extra[:3]}")
    for name, module in modules.items():
        for suffix in ("lora_A", "lora_B"):
            parameter = getattr(module, suffix)
            parameter.data.copy_(state[f"{name}.{suffix}"].to(parameter.device))


class MediaCache:
    def __init__(self, protocol: QwenProtocol) -> None:
        self.protocol = protocol
        self.values: dict[str, tuple[Any, Any, Any]] = {}

    def get(self, row: dict[str, Any], process_mm_info: Any) -> tuple[Any, Any, Any]:
        media_path = str(row["media_path"])
        if media_path not in self.values:
            conversation = _conversation(row, self.protocol)
            self.values[media_path] = process_mm_info(
                conversation,
                use_audio_in_video=True,
            )
        return self.values[media_path]

    def clear(self) -> None:
        self.values.clear()


def _conversation(
    row: dict[str, Any],
    protocol: QwenProtocol,
    answer: str | None = None,
) -> list[dict[str, Any]]:
    content = [
        {
            "type": "video",
            "video": str(row["media_path"]),
            "fps": protocol.fps,
            "min_frames": protocol.min_frames,
            "max_frames": protocol.max_frames,
            "total_pixels": protocol.total_pixels,
        },
        {"type": "text", "text": protocol.render_prompt(row)},
    ]
    conversation: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        response = str(answer).strip()
        if not response:
            raise ValueError(f"{row.get('qa_id')}: empty response target")
        conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )
    return conversation


def _find_last_subsequence(sequence: list[int], target: list[int]) -> int:
    for start in range(len(sequence) - len(target), -1, -1):
        if sequence[start : start + len(target)] == target:
            return start
    return -1


class QwenOmniRuntime:
    """Concrete runtime consumed by ``train_sde.py`` and ``train_cctdh.py``."""

    def __init__(
        self,
        *,
        stage: str,
        config: dict[str, Any],
        model_path: str | Path,
        protocol_config: str | Path = "baseline/config.yaml",
        protocol_name: str = "music_avqa_42",
        sde_adapter: str | Path | None = None,
        teacher_bundle: str | Path | None = None,
        dtype: str = "bfloat16",
        attention: str = "sdpa",
        device: str = "cuda",
        learning_rate: float | None = None,
        weight_decay: float | None = None,
    ) -> None:
        if stage not in {"sde", "cctdh"}:
            raise ValueError(f"unsupported Qwen training stage: {stage}")
        try:
            from qwen_omni_utils import process_mm_info
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as error:
            raise ImportError(
                "Qwen runtime requires transformers and qwen-omni-utils"
            ) from error

        self.stage = stage
        self.config = config
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_config_hash = _sha256(self.model_path / "config.json")
        self.protocol_config_path = Path(protocol_config).expanduser().resolve()
        self.protocol_config_hash = _protocol_sha256(
            self.protocol_config_path,
            protocol_name,
        )
        self.protocol = QwenProtocol.from_yaml(
            self.protocol_config_path,
            protocol_name,
        )
        self.process_mm_info = process_mm_info
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            torch_dtype=_torch_dtype(dtype),
            device_map=device,
            attn_implementation=attention,
            enable_audio_output=False,
        )
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.model.requires_grad_(False)
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            use_fast=False,
        )
        self.media_cache = MediaCache(self.protocol)
        self.sde_modules: dict[str, LoRALinear]
        self.modules: dict[str, Any]
        self.teacher: dict[str, dict[str, torch.Tensor]] = {}
        self.hidden_layer: int | None = None
        self._captured: dict[str, torch.Tensor] = {}
        self._hidden_hook: Any = None
        self.parent_adapter_hash: str | None = None
        self.residual_adapter_hash: str | None = None
        self.teacher_bundle_hash: str | None = None

        sde_config = config["sde"]["lora"]
        lora_keys = ("layers", "rank", "alpha", "dropout")
        self.sde_modules = install_sde_lora(
            self.model,
            **{key: sde_config[key] for key in lora_keys},
        )
        if stage == "sde":
            self.modules = self.sde_modules
        else:
            if sde_adapter is None:
                raise ValueError("CC-TDH Qwen runtime requires the SDE adapter")
            sde_adapter_file = _adapter_file(sde_adapter)
            payload = torch.load(
                sde_adapter_file,
                map_location="cpu",
                weights_only=False,
            )
            self._validate_sde_payload(payload)
            load_adapter_state(self.sde_modules, payload["state_dict"])
            self.parent_adapter_hash = _sha256(sde_adapter_file)
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            residual_config = config["cctdh"]["residual_lora"]
            self.modules = install_residual_lora(
                self.model,
                **{key: residual_config[key] for key in lora_keys},
            )
            if teacher_bundle is None:
                raise ValueError("CC-TDH Qwen runtime requires a teacher bundle")
            teacher_bundle = Path(teacher_bundle).expanduser().resolve()
            teacher_payload = torch.load(
                teacher_bundle,
                map_location="cpu",
                weights_only=False,
            )
            self.teacher_bundle_hash = _sha256(teacher_bundle)
            self.teacher = teacher_payload["teachers"]
            self.hidden_layer = int(
                teacher_payload.get(
                    "hidden_layer",
                    config["teacher"]["hidden_layer"],
                )
            )
            self._hidden_hook = self.model.thinker.model.layers[
                self.hidden_layer
            ].register_forward_hook(self._capture_hidden)

        for module in self.modules.values():
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        parameters = self.trainable_parameters()
        optimizer_config = config[stage].get("optimizer", {})
        lr = float(
            learning_rate
            if learning_rate is not None
            else optimizer_config.get("learning_rate", 5e-5)
        )
        decay = float(
            weight_decay
            if weight_decay is not None
            else optimizer_config.get("weight_decay", 0.0)
        )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=decay,
        )

    def _validate_sde_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("stage") != "sde":
            raise ValueError("parent adapter is not an SDE adapter")
        if payload.get("model_config_sha256") != self.model_config_hash:
            raise ValueError("SDE adapter was built for a different model config")
        if payload.get("protocol") != self.protocol.name:
            raise ValueError("SDE adapter uses a different answer protocol")
        if payload.get("protocol_config_sha256") != self.protocol_config_hash:
            raise ValueError("SDE adapter uses a different prompt configuration")
        if payload.get("config") != self.config["sde"]["lora"]:
            raise ValueError("SDE adapter LoRA configuration differs")

    def _capture_hidden(self, _module: Any, _args: Any, output: Any) -> None:
        self._captured["hidden"] = output[0] if isinstance(output, tuple) else output

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for module in self.modules.values()
            for parameter in (module.lora_A, module.lora_B)
        ]

    def _move(self, inputs: Any) -> Any:
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                value = value.to(self.device, non_blocking=True)
                if value.is_floating_point():
                    value = value.to(self.dtype)
                inputs[key] = value
        return inputs

    def _media(self, row: dict[str, Any]) -> tuple[Any, Any, Any]:
        return self.media_cache.get(row, self.process_mm_info)

    def prepare_inference(self, row: dict[str, Any]) -> dict[str, Any]:
        conversation = _conversation(row, self.protocol)
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audio, image, video = self._media(row)
        inputs = self.processor(
            text=[text],
            audio=audio or None,
            images=image or None,
            videos=video or None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        return self._move(inputs)

    def prepare_training(
        self,
        row: dict[str, Any],
        answer: str,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        response = str(answer).strip()
        if not response:
            raise ValueError(f"{row.get('qa_id')}: empty target")
        conversation = _conversation(row, self.protocol, response)
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
        )
        audio, image, video = self._media(row)
        inputs = self.processor(
            text=[text],
            audio=audio or None,
            images=image or None,
            videos=video or None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        valid_length = int(inputs.attention_mask[0].sum())
        token_ids = inputs.input_ids[0, :valid_length].tolist()
        answer_ids = self.processor.tokenizer.encode(
            response,
            add_special_tokens=False,
        )
        start = _find_last_subsequence(token_ids, answer_ids)
        if start < 0:
            raise RuntimeError(
                f"{row.get('qa_id')}: response tokens absent from chat template"
            )
        end = start + len(answer_ids)
        if end >= valid_length:
            raise RuntimeError(f"{row.get('qa_id')}: truncated response boundary")
        labels = torch.full_like(inputs.input_ids, IGNORE_INDEX)
        labels[0, start:end] = inputs.input_ids[0, start:end]
        labels[inputs.attention_mask == 0] = IGNORE_INDEX
        if int((labels != IGNORE_INDEX).sum()) != len(answer_ids):
            raise RuntimeError("response-only mask cardinality mismatch")
        inputs["labels"] = labels
        return self._move(inputs), {
            "response_start": start,
            "response_end": end,
            "answer_tokens": len(answer_ids),
        }

    def response_nll(
        self,
        rows: list[dict[str, Any]],
        target_field: str,
    ) -> torch.Tensor:
        losses = []
        for row in rows:
            inputs, _ = self.prepare_training(row, str(row[target_field]))
            output = self.model.thinker(
                **inputs,
                use_audio_in_video=True,
                use_cache=False,
                return_dict=True,
            )
            losses.append(output.loss)
            self.media_cache.clear()
        return torch.stack(losses)

    def retention_gradients(
        self,
        rows: list[dict[str, Any]],
        target_field: str,
        batch_size: int,
    ) -> torch.Tensor:
        parameters = self.trainable_parameters()
        gradients = []
        for start in range(0, len(rows), batch_size):
            loss = self.response_nll(rows[start : start + batch_size], target_field).mean()
            gradients.append(flat_autograd(loss, parameters).detach().cpu())
        return torch.stack(gradients)

    def detail_loss(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        if self.stage != "cctdh" or self.hidden_layer is None:
            raise RuntimeError("detail loss is available only in CC-TDH")
        losses = []
        for row in rows:
            self._captured.clear()
            answer = str(_row_value(row, "gold_response", row["gold_answer"]))
            inputs, metadata = self.prepare_training(row, answer)
            self.model.thinker(
                **inputs,
                use_audio_in_video=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = self._captured["hidden"][
                0, metadata["response_start"] - 1
            ].float()
            expert = str(_row_value(row, "selected_expert", row.get("expert_name")))
            if expert not in self.teacher:
                raise KeyError(f"teacher bundle has no route for {expert!r}")
            entry = self.teacher[expert]
            weight = entry["weight"].to(hidden.device, dtype=torch.float32)
            predicted = F.normalize(hidden @ weight, dim=-1, eps=1e-8)
            target = torch.as_tensor(
                row["detail_target"],
                device=hidden.device,
                dtype=torch.float32,
            )
            target = F.normalize(target, dim=-1, eps=1e-8)
            losses.append(1.0 - torch.dot(predicted, target))
            self.media_cache.clear()
        return torch.stack(losses)

    @torch.inference_mode()
    def extract_hidden(
        self,
        rows: list[dict[str, Any]],
        *,
        layer: int,
        target_field: str = "gold_response",
    ) -> list[dict[str, Any]]:
        """Extract frozen-parent answer-boundary hidden states for Eq. 8."""
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = self.model.thinker.model.layers[int(layer)].register_forward_hook(hook)
        records = []
        try:
            for row in rows:
                captured.clear()
                target = str(
                    _row_value(
                        row,
                        target_field,
                        _row_value(row, "gold_answer"),
                    )
                )
                inputs, metadata = self.prepare_training(row, target)
                self.model.thinker(
                    **inputs,
                    use_audio_in_video=True,
                    use_cache=False,
                    return_dict=True,
                )
                value = captured["hidden"][
                    0, metadata["response_start"] - 1
                ].detach().float().cpu()
                records.append(
                    {
                        "qa_id": str(row["qa_id"]),
                        "source_id": str(row["source_id"]),
                        "parent_hidden": value.tolist(),
                        "hidden_layer": int(layer),
                        "answer_position": "response_start_minus_one",
                    }
                )
                self.media_cache.clear()
        finally:
            handle.remove()
        return records

    @torch.inference_mode()
    def generate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs = []
        self.model.eval()
        for row in rows:
            inputs = self.prepare_inference(row)
            generated = self.model.thinker.generate(
                **inputs,
                use_audio_in_video=True,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.protocol.max_new_tokens,
                use_cache=True,
            )
            prompt_length = inputs.input_ids.shape[1]
            raw = self.processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            outputs.append(
                {
                    "qa_id": str(row["qa_id"]),
                    "source_id": str(row["source_id"]),
                    "raw_response": raw,
                    "answer": self.protocol.normalize_answer(raw),
                }
            )
            self.media_cache.clear()
        return outputs

    def load_sde_for_inference(self, adapter: str | Path) -> None:
        adapter_file = _adapter_file(adapter)
        payload = torch.load(
            adapter_file,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_sde_payload(payload)
        load_adapter_state(self.sde_modules, payload["state_dict"])
        self.parent_adapter_hash = _sha256(adapter_file)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def load_cctdh_for_inference(self, adapter: str | Path) -> None:
        """Stack a frozen residual adapter on the already-loaded SDE parent."""
        if self.parent_adapter_hash is None:
            raise RuntimeError("load the SDE parent before the CC-TDH adapter")
        adapter_file = _adapter_file(adapter)
        payload = torch.load(
            adapter_file,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("stage") != "cctdh":
            raise ValueError("residual adapter is not a CC-TDH adapter")
        if payload.get("parent_stage") != "sde":
            raise ValueError("CC-TDH adapter does not declare an SDE parent")
        if payload.get("parent_adapter_sha256") != self.parent_adapter_hash:
            raise ValueError("CC-TDH adapter was trained from a different SDE parent")
        if payload.get("model_config_sha256") != self.model_config_hash:
            raise ValueError("CC-TDH adapter was built for a different model config")
        if payload.get("protocol") != self.protocol.name:
            raise ValueError("CC-TDH adapter uses a different answer protocol")
        if payload.get("protocol_config_sha256") != self.protocol_config_hash:
            raise ValueError("CC-TDH adapter uses a different prompt configuration")
        if payload.get("config") != self.config["cctdh"]["residual_lora"]:
            raise ValueError("CC-TDH residual LoRA configuration differs")
        residual_config = self.config["cctdh"]["residual_lora"]
        lora_keys = ("layers", "rank", "alpha", "dropout")
        self.modules = install_residual_lora(
            self.model,
            **{key: residual_config[key] for key in lora_keys},
        )
        load_adapter_state(self.modules, payload["state_dict"])
        self.residual_adapter_hash = _sha256(adapter_file)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def merge_for_deployment(self) -> None:
        """Merge the loaded SDE and residual deltas into the Base linear weights."""
        if not self.modules or not all(
            isinstance(module, ResidualLoRALinear)
            for module in self.modules.values()
        ):
            raise RuntimeError("deployment merge requires both SDE and CC-TDH adapters")
        owners = {
            name: (owner, attribute)
            for name, owner, attribute in _target_modules(
                self.model,
                self.config["cctdh"]["residual_lora"]["layers"],
            )
        }
        with torch.no_grad():
            for name, residual in self.modules.items():
                parent = residual.parent
                base = parent.base
                sde_delta = parent.lora_B.float() @ parent.lora_A.float()
                residual_delta = residual.lora_B.float() @ residual.lora_A.float()
                merged = (
                    base.weight.float()
                    + parent.scaling * sde_delta
                    + residual.scaling * residual_delta
                )
                base.weight.copy_(merged.to(base.weight.dtype))
                owner, attribute = owners[name]
                setattr(owner, attribute, base)
        self.sde_modules = {}
        self.modules = {}

    def save_deployed(self, output_dir: str | Path) -> None:
        """Write a standalone expert-free model after merging both adapters."""
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"deployment output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        self.merge_for_deployment()
        self.model.save_pretrained(output, safe_serialization=True)
        self.processor.save_pretrained(output)
        (output / "deployment_manifest.json").write_text(
            json.dumps(
                {
                    "base_model_config_sha256": self.model_config_hash,
                    "protocol": self.protocol.name,
                    "protocol_config_sha256": self.protocol_config_hash,
                    "parent_adapter_sha256": self.parent_adapter_hash,
                    "residual_adapter_sha256": self.residual_adapter_hash,
                    "experts_required_at_inference": False,
                    "teacher_required_at_inference": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": self.stage,
            "state_dict": adapter_state(self.modules),
            "model_path": str(self.model_path),
            "model_config_sha256": self.model_config_hash,
            "protocol": self.protocol.name,
            "protocol_config_sha256": self.protocol_config_hash,
            "config": (
                self.config["sde"]["lora"]
                if self.stage == "sde"
                else self.config["cctdh"]["residual_lora"]
            ),
        }
        if self.stage == "cctdh":
            payload["parent_stage"] = "sde"
            payload["parent_adapter_sha256"] = self.parent_adapter_hash
            payload["teacher_bundle_sha256"] = self.teacher_bundle_hash
        torch.save(payload, output_dir / "adapter.pt")
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "state_dict"
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._hidden_hook is not None:
            self._hidden_hook.remove()
            self._hidden_hook = None
        self.media_cache.clear()


def build_runtime(
    *,
    stage: str,
    config: dict[str, Any],
    **runtime_args: Any,
) -> QwenOmniRuntime:
    """Factory used as ``sde_cctdh.qwen:build_runtime``."""
    seed = int(runtime_args.pop("seed", config.get("seed", 18)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return QwenOmniRuntime(stage=stage, config=config, **runtime_args)
