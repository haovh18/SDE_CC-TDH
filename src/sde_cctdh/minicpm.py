"""MiniCPM-o-4.5-GPTQ runtime for the SDE -> CC-TDH method."""
from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen import (
    IGNORE_INDEX,
    QwenProtocol,
    _adapter_file,
    _find_last_subsequence,
    _protocol_sha256,
    _row_value,
    _sha256,
    _torch_dtype,
)
from .training import flat_autograd


def _linear_size(module: nn.Module, name: str) -> int:
    """Read dimensions from torch, AutoGPTQ, or GPTQModel linear modules."""
    for attribute in (name, name.replace("_features", "features")):
        value = getattr(module, attribute, None)
        if value is not None:
            return int(value)
    raise TypeError(f"GPTQ projection {type(module)!r} has no {name} metadata")


def _module_device(module: nn.Module) -> torch.device:
    for value in list(module.parameters()) + list(module.buffers()):
        return value.device
    raise TypeError(f"cannot determine device for {type(module)!r}")


class GPTQLoRALinear(nn.Module):
    """LoRA wrapper that leaves the W4A16 GPTQ projection quantized."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        device = _module_device(base)
        in_features = _linear_size(base, "in_features")
        out_features = _linear_size(base, "out_features")
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        update = F.linear(
            F.linear(self.dropout(value).to(self.lora_A.dtype), self.lora_A),
            self.lora_B,
        )
        return base + update.to(base.dtype) * self.scaling


class ResidualGPTQLoRALinear(nn.Module):
    """A trainable residual LoRA above a frozen SDE GPTQ-LoRA parent."""

    def __init__(
        self,
        parent: GPTQLoRALinear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        device = parent.lora_A.device
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                parent.lora_A.shape[1],
                device=device,
                dtype=dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                parent.lora_B.shape[0],
                self.rank,
                device=device,
                dtype=dtype,
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
    blocks = model.llm.model.layers
    for layer_index in layers:
        layer = int(layer_index)
        attention = blocks[layer].self_attn
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield (
                f"llm.model.layers.{layer}.self_attn.{projection}",
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
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, GPTQLoRALinear]:
    modules: dict[str, GPTQLoRALinear] = {}
    for name, owner, attribute in _target_modules(model, layers):
        base = getattr(owner, attribute)
        if isinstance(base, (GPTQLoRALinear, ResidualGPTQLoRALinear)):
            raise TypeError(f"MiniCPM LoRA target is already wrapped: {name}")
        wrapped = GPTQLoRALinear(
            base,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=dtype,
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
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, ResidualGPTQLoRALinear]:
    modules: dict[str, ResidualGPTQLoRALinear] = {}
    for name, owner, attribute in _target_modules(model, layers):
        parent = getattr(owner, attribute)
        if not isinstance(parent, GPTQLoRALinear):
            raise TypeError(f"SDE parent LoRA is missing at {name}")
        wrapped = ResidualGPTQLoRALinear(
            parent,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=dtype,
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
            parameter.data.copy_(
                state[f"{name}.{suffix}"].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def response_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal, response-only, response-length-normalized negative log likelihood."""
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    active = shifted_labels.ne(IGNORE_INDEX)
    token_count = active.sum(dim=1)
    if bool(token_count.eq(0).any()):
        raise RuntimeError("response mask contains no causally predicted token")
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    return (token_losses * active).sum(dim=1) / token_count


def _move_nested(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        value = value.to(device, non_blocking=True)
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, list):
        return [_move_nested(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _move_nested(item, device, dtype) for key, item in value.items()}
    return value


class MiniCPMMediaCache:
    def __init__(self, normalize_content: Any, preprocessing: dict[str, Any]) -> None:
        self.normalize_content = normalize_content
        self.preprocessing = preprocessing
        self.values: dict[str, tuple[list[Any], list[Any]]] = {}

    def get(self, row: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        path = str(row["media_path"])
        if path not in self.values:
            video = {
                "type": "video_url",
                "video_url": {
                    "url": path,
                    "use_audio": bool(self.preprocessing["use_audio"]),
                    "stack_frames": int(self.preprocessing["stack_frames"]),
                    "use_ffmpeg": bool(self.preprocessing.get("use_ffmpeg", False)),
                },
            }
            content = self.normalize_content([video])
            try:
                from PIL import Image
            except ImportError as error:
                raise ImportError("MiniCPM video preprocessing requires Pillow") from error
            images = [item for item in content if isinstance(item, Image.Image)]
            audios = [item for item in content if isinstance(item, np.ndarray)]
            if not images:
                raise RuntimeError(f"MiniCPM preprocessing returned no video frames: {path}")
            if self.preprocessing["use_audio"] and not audios:
                raise RuntimeError(f"MiniCPM preprocessing returned no audio: {path}")
            self.values[path] = (images, audios)
        return self.values[path]

    def clear(self) -> None:
        self.values.clear()


class MiniCPMRuntime:
    """Runtime interface consumed by the common SDE and CC-TDH trainers."""

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
        device: str = "auto",
        learning_rate: float | None = None,
        weight_decay: float | None = None,
    ) -> None:
        if stage not in {"sde", "cctdh"}:
            raise ValueError(f"unsupported MiniCPM training stage: {stage}")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "MiniCPM runtime requires transformers and AutoGPTQ/GPTQModel"
            ) from error

        self.stage = stage
        self.config = config
        self.adapter_dtype = _torch_dtype(dtype)
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_config_hash = _sha256(self.model_path / "config.json")
        self.protocol_config_path = Path(protocol_config).expanduser().resolve()
        self.protocol_config_hash = _protocol_sha256(
            self.protocol_config_path,
            protocol_name,
        )
        self.protocol = QwenProtocol.from_yaml(self.protocol_config_path, protocol_name)
        model_config = AutoConfig.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        if getattr(model_config, "quantization_config", None):
            model_config.quantization_config = dict(model_config.quantization_config)
            model_config.quantization_config["block_name_to_quantize"] = "llm.model.layers"
        model_config.init_vision = True
        model_config.init_audio = True
        model_config.init_tts = True
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
            config=model_config,
            attn_implementation=attention,
            torch_dtype=self.adapter_dtype,
            device_map=device,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.prepare_processor(tokenizer=self.tokenizer)
        self.processor = self.model.processor
        remote_module = importlib.import_module(self.model.__class__.__module__)
        preprocessing = {
            "max_slice_nums": 1,
            "max_input_length": 40960,
            "stack_frames": 1,
            "use_audio": True,
            "use_ffmpeg": False,
            **config.get("preprocessing", {}),
        }
        self.preprocessing = preprocessing
        self.media_cache = MiniCPMMediaCache(remote_module.normalize_content, preprocessing)
        self.teacher: dict[str, dict[str, torch.Tensor]] = {}
        self.hidden_layer: int | None = None
        self._captured: dict[str, torch.Tensor] = {}
        self._hidden_hook: Any = None
        self.parent_adapter_hash: str | None = None
        self.residual_adapter_hash: str | None = None
        self.teacher_bundle_hash: str | None = None

        lora_keys = ("layers", "rank", "alpha", "dropout")
        sde_config = config["sde"]["lora"]
        self.sde_modules = install_sde_lora(
            self.model,
            dtype=self.adapter_dtype,
            **{key: sde_config[key] for key in lora_keys},
        )
        if stage == "sde":
            self.modules: dict[str, Any] = self.sde_modules
        else:
            if sde_adapter is None:
                raise ValueError("CC-TDH MiniCPM runtime requires the SDE adapter")
            self._load_sde(sde_adapter)
            residual_config = config["cctdh"]["residual_lora"]
            self.modules = install_residual_lora(
                self.model,
                dtype=self.adapter_dtype,
                **{key: residual_config[key] for key in lora_keys},
            )
            if teacher_bundle is None:
                raise ValueError("CC-TDH MiniCPM runtime requires a teacher bundle")
            teacher_path = Path(teacher_bundle).expanduser().resolve()
            teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
            self.teacher_bundle_hash = _sha256(teacher_path)
            self.teacher = teacher_payload["teachers"]
            self.hidden_layer = int(
                teacher_payload.get("hidden_layer", config["teacher"]["hidden_layer"])
            )
            self._hidden_hook = self.model.llm.model.layers[
                self.hidden_layer
            ].register_forward_hook(self._capture_hidden)

        for module in self.modules.values():
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        optimizer_config = config[stage].get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=float(
                learning_rate
                if learning_rate is not None
                else optimizer_config.get("learning_rate", 5e-5)
            ),
            weight_decay=float(
                weight_decay
                if weight_decay is not None
                else optimizer_config.get("weight_decay", 0.0)
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.sde_modules[next(iter(self.sde_modules))].lora_A.device

    @property
    def dtype(self) -> torch.dtype:
        return self.adapter_dtype

    def _capture_hidden(self, _module: Any, _args: Any, output: Any) -> None:
        self._captured["hidden"] = output[0] if isinstance(output, tuple) else output

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for module in self.modules.values()
            for parameter in (module.lora_A, module.lora_B)
        ]

    def _prepare(
        self,
        row: dict[str, Any],
        answer: str | None,
    ) -> tuple[dict[str, Any], dict[str, int] | None]:
        images, audios = self.media_cache.get(row)
        user_content = ["<image>./</image>"] * len(images)
        user_content += ["<audio>./</audio>"] * len(audios)
        user_content.append(self.protocol.render_prompt(row))
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "\n".join(user_content)}
        ]
        if answer is not None:
            response = str(answer).strip()
            if not response:
                raise ValueError(f"{row.get('qa_id')}: empty response target")
            messages.append({"role": "assistant", "content": response})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=answer is None,
            use_tts_template=bool(audios),
            enable_thinking=False,
        )
        data = self.processor(
            [prompt],
            [images],
            [audios],
            [[0] * len(audios)],
            max_slice_nums=int(self.preprocessing["max_slice_nums"]),
            return_tensors="pt",
            max_length=int(self.preprocessing["max_input_length"]),
        )
        data.pop("image_sizes", None)
        data = _move_nested(dict(data), self.device, self.dtype)
        if answer is None:
            return data, None
        valid_length = int(data["attention_mask"][0].sum())
        token_ids = data["input_ids"][0, :valid_length].tolist()
        answer_ids = self.tokenizer.encode(response, add_special_tokens=False)
        start = _find_last_subsequence(token_ids, answer_ids)
        if start < 1:
            raise RuntimeError(
                f"{row.get('qa_id')}: response tokens absent from MiniCPM chat template"
            )
        end = start + len(answer_ids)
        if end >= valid_length:
            raise RuntimeError(f"{row.get('qa_id')}: truncated response boundary")
        labels = torch.full_like(data["input_ids"], IGNORE_INDEX)
        labels[0, start:end] = data["input_ids"][0, start:end]
        labels[data["attention_mask"] == 0] = IGNORE_INDEX
        if int(labels.ne(IGNORE_INDEX).sum()) != len(answer_ids):
            raise RuntimeError("response-only mask cardinality mismatch")
        return data, {
            "response_start": start,
            "response_end": end,
            "answer_tokens": len(answer_ids),
            "labels": labels,
        }

    def prepare_training(
        self,
        row: dict[str, Any],
        answer: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        data, metadata = self._prepare(row, answer)
        assert metadata is not None
        return data, metadata

    def response_nll(
        self,
        rows: list[dict[str, Any]],
        target_field: str,
    ) -> torch.Tensor:
        losses = []
        for row in rows:
            data, metadata = self.prepare_training(row, str(row[target_field]))
            output = self.model(data, use_cache=False, return_dict=True)
            losses.append(response_token_nll(output.logits, metadata["labels"])[0])
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
            data, metadata = self.prepare_training(row, answer)
            self.model(data, use_cache=False, return_dict=True)
            hidden = self._captured["hidden"][
                0, metadata["response_start"] - 1
            ].float()
            expert = str(_row_value(row, "selected_expert", row.get("expert_name")))
            if expert not in self.teacher:
                raise KeyError(f"teacher bundle has no route for {expert!r}")
            weight = self.teacher[expert]["weight"].to(
                hidden.device,
                dtype=torch.float32,
            )
            predicted = F.normalize(hidden @ weight, dim=-1, eps=1e-8)
            target = F.normalize(
                torch.as_tensor(
                    row["detail_target"],
                    device=hidden.device,
                    dtype=torch.float32,
                ),
                dim=-1,
                eps=1e-8,
            )
            losses.append(1.0 - torch.dot(predicted, target))
        return torch.stack(losses)

    @torch.inference_mode()
    def extract_hidden(
        self,
        rows: list[dict[str, Any]],
        *,
        layer: int,
        target_field: str = "gold_response",
    ) -> list[dict[str, Any]]:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = self.model.llm.model.layers[int(layer)].register_forward_hook(hook)
        records = []
        try:
            for row in rows:
                captured.clear()
                answer = str(
                    _row_value(row, target_field, _row_value(row, "gold_answer"))
                )
                data, metadata = self.prepare_training(row, answer)
                self.model(data, use_cache=False, return_dict=True)
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
        finally:
            handle.remove()
        return records

    @torch.inference_mode()
    def generate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs = []
        self.model.eval()
        for row in rows:
            media = {
                "type": "video_url",
                "video_url": {
                    "url": str(row["media_path"]),
                    "use_audio": bool(self.preprocessing["use_audio"]),
                    "stack_frames": int(self.preprocessing["stack_frames"]),
                    "use_ffmpeg": bool(self.preprocessing.get("use_ffmpeg", False)),
                },
            }
            messages = [
                {
                    "role": "user",
                    "content": [media, {"type": "text", "text": self.protocol.render_prompt(row)}],
                }
            ]
            result = self.model.chat(
                msgs=messages,
                tokenizer=self.tokenizer,
                max_new_tokens=self.protocol.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_tts_template=False,
                enable_thinking=False,
                generate_audio=False,
                max_slice_nums=int(self.preprocessing["max_slice_nums"]),
                max_inp_length=int(self.preprocessing["max_input_length"]),
            )
            raw = str(result[0] if isinstance(result, tuple) else result).strip()
            outputs.append(
                {
                    "qa_id": str(row["qa_id"]),
                    "source_id": str(row["source_id"]),
                    "raw_response": raw,
                    "answer": self.protocol.normalize_answer(raw),
                }
            )
        return outputs

    def _validate_sde_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("stage") != "sde":
            raise ValueError("parent adapter is not an SDE adapter")
        if payload.get("runtime") != "minicpm_o_4_5_gptq":
            raise ValueError("parent adapter is not a MiniCPM-GPTQ adapter")
        if payload.get("model_config_sha256") != self.model_config_hash:
            raise ValueError("SDE adapter was built for a different model config")
        if payload.get("protocol") != self.protocol.name:
            raise ValueError("SDE adapter uses a different answer protocol")
        if payload.get("protocol_config_sha256") != self.protocol_config_hash:
            raise ValueError("SDE adapter uses a different prompt configuration")
        if payload.get("config") != self.config["sde"]["lora"]:
            raise ValueError("SDE adapter LoRA configuration differs")

    def _load_sde(self, adapter: str | Path) -> None:
        adapter_file = _adapter_file(adapter)
        payload = torch.load(adapter_file, map_location="cpu", weights_only=False)
        self._validate_sde_payload(payload)
        load_adapter_state(self.sde_modules, payload["state_dict"])
        self.parent_adapter_hash = _sha256(adapter_file)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def load_sde_for_inference(self, adapter: str | Path) -> None:
        self._load_sde(adapter)

    def load_cctdh_for_inference(self, adapter: str | Path) -> None:
        if self.parent_adapter_hash is None:
            raise RuntimeError("load the SDE parent before the CC-TDH adapter")
        adapter_file = _adapter_file(adapter)
        payload = torch.load(adapter_file, map_location="cpu", weights_only=False)
        checks = {
            "stage": "cctdh",
            "runtime": "minicpm_o_4_5_gptq",
            "parent_stage": "sde",
            "parent_adapter_sha256": self.parent_adapter_hash,
            "model_config_sha256": self.model_config_hash,
            "protocol": self.protocol.name,
            "protocol_config_sha256": self.protocol_config_hash,
            "config": self.config["cctdh"]["residual_lora"],
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise ValueError(f"CC-TDH adapter has incompatible {key}")
        residual_config = self.config["cctdh"]["residual_lora"]
        keys = ("layers", "rank", "alpha", "dropout")
        self.modules = install_residual_lora(
            self.model,
            dtype=self.adapter_dtype,
            **{key: residual_config[key] for key in keys},
        )
        load_adapter_state(self.modules, payload["state_dict"])
        self.residual_adapter_hash = _sha256(adapter_file)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "stage": self.stage,
            "runtime": "minicpm_o_4_5_gptq",
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
            payload.update(
                parent_stage="sde",
                parent_adapter_sha256=self.parent_adapter_hash,
                teacher_bundle_sha256=self.teacher_bundle_hash,
            )
        torch.save(payload, output_dir / "adapter.pt")
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {key: value for key, value in payload.items() if key != "state_dict"},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def save_deployed(self, output_dir: str | Path) -> None:
        """Export the two inference-only adapters without duplicating GPTQ weights."""
        if self.parent_adapter_hash is None or self.residual_adapter_hash is None:
            raise RuntimeError("deployment export requires loaded SDE and CC-TDH adapters")
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"deployment output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        sde_payload = {
            "stage": "sde",
            "runtime": "minicpm_o_4_5_gptq",
            "state_dict": adapter_state(self.sde_modules),
            "model_config_sha256": self.model_config_hash,
            "protocol": self.protocol.name,
            "protocol_config_sha256": self.protocol_config_hash,
            "config": self.config["sde"]["lora"],
        }
        sde_export = output / "sde_adapter.pt"
        torch.save(sde_payload, sde_export)
        exported_parent_hash = _sha256(sde_export)
        cctdh_payload = {
            "stage": "cctdh",
            "runtime": "minicpm_o_4_5_gptq",
            "parent_stage": "sde",
            "parent_adapter_sha256": exported_parent_hash,
            "state_dict": adapter_state(self.modules),
            "model_config_sha256": self.model_config_hash,
            "protocol": self.protocol.name,
            "protocol_config_sha256": self.protocol_config_hash,
            "config": self.config["cctdh"]["residual_lora"],
        }
        torch.save(cctdh_payload, output / "cctdh_adapter.pt")
        (output / "deployment_manifest.json").write_text(
            json.dumps(
                {
                    "base_model": str(self.model_path),
                    "base_model_config_sha256": self.model_config_hash,
                    "load_order": ["sde_adapter.pt", "cctdh_adapter.pt"],
                    "parent_adapter_sha256": exported_parent_hash,
                    "residual_adapter_sha256": self.residual_adapter_hash,
                    "experts_required_at_inference": False,
                    "teacher_required_at_inference": False,
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


def create_runtime(
    *,
    stage: str,
    config: dict[str, Any],
    **runtime_args: Any,
) -> MiniCPMRuntime:
    """Factory used as ``sde_cctdh.minicpm:create_runtime``."""
    seed = int(runtime_args.pop("seed", config.get("seed", 18)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return MiniCPMRuntime(stage=stage, config=config, **runtime_args)


build_runtime = create_runtime
