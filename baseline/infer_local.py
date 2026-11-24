#!/usr/bin/env python3
"""Run Qwen2.5-Omni or MiniCPM-o baselines on a shared JSONL manifest."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.protocol import append_jsonl, load_config, parse_prediction, read_jsonl, render_prompt, validate_row


def move_inputs(inputs: Any, model: Any, torch: Any) -> Any:
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            value = value.to(model.device, non_blocking=True)
            if value.is_floating_point():
                value = value.to(model.dtype)
            inputs[key] = value
    return inputs


class QwenRunner:
    def __init__(self, model_path: Path, attention: str) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=attention,
            enable_audio_output=False,
        )
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.model.requires_grad_(False)
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), local_files_only=True)

    def __call__(self, row: dict[str, Any], prompt: str, generation: dict[str, Any]) -> str:
        from qwen_omni_utils import process_mm_info

        media = generation["qwen_video"]
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": row["media_path"],
                        "fps": float(media["fps"]),
                        "min_frames": int(media["min_frames"]),
                        "max_frames": int(media["max_frames"]),
                        "total_pixels": int(media["total_pixels"]),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, image, video = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(
            text=[text],
            audio=audio or None,
            images=image or None,
            videos=video or None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = move_inputs(inputs, self.model, self.torch)
        max_tokens = int(generation["max_new_tokens"][row["dataset"]])
        generator = getattr(self.model, "thinker", self.model)
        with self.torch.inference_mode():
            generated = generator.generate(
                **inputs,
                use_audio_in_video=True,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True,
            )
        prefix = int(inputs["input_ids"].shape[1])
        return self.processor.batch_decode(
            generated[:, prefix:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class MiniCPMRunner:
    def __init__(self, model_path: Path, attention: str) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        if getattr(config, "quantization_config", None):
            config.quantization_config = dict(config.quantization_config)
            config.quantization_config["block_name_to_quantize"] = "llm.model.layers"
        config.init_vision = True
        config.init_audio = True
        config.init_tts = True
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            config=config,
            attn_implementation=attention,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )

    def __call__(self, row: dict[str, Any], prompt: str, generation: dict[str, Any]) -> str:
        media = {
            "type": "video_url",
            "video_url": {"url": row["media_path"], "use_audio": True, "stack_frames": 1},
        }
        messages = [{"role": "user", "content": [media, {"type": "text", "text": prompt}]}]
        settings = generation["minicpm"]
        with self.torch.inference_mode():
            output = self.model.chat(
                msgs=messages,
                tokenizer=self.tokenizer,
                max_new_tokens=int(generation["max_new_tokens"][row["dataset"]]),
                do_sample=False,
                use_tts_template=False,
                enable_thinking=False,
                generate_audio=False,
                max_slice_nums=int(settings["max_slice_nums"]),
                max_inp_length=int(settings["max_input_length"]),
            )
        return str(output[0] if isinstance(output, tuple) else output)


def resolve_model_path(config: dict[str, Any], model_name: str, override: Path | None) -> Path:
    if override is not None:
        path = override
    else:
        path = Path(__file__).resolve().parent.parent / config["models"][model_name]["local_dir"]
    path = path.expanduser().resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"incomplete model directory: {path}; "
            f"run model/download_models.py avllm {model_name}"
        )
    return path


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["models"][args.model]
    if spec["family"] not in {"qwen2_5_omni", "minicpm_o_45"}:
        raise ValueError(f"{args.model} is not a local inference model")
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        validate_row(row)
    model_path = resolve_model_path(config, args.model, args.model_path)
    runner = (
        QwenRunner(model_path, args.attention)
        if spec["family"] == "qwen2_5_omni"
        else MiniCPMRunner(model_path, args.attention)
    )
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {str(row["qa_id"]) for row in existing if row.get("terminal_status") == "success"}
    if len(completed) != len(existing):
        duplicate_or_failed = len(existing) - len(completed)
        print(f"resume: retaining {duplicate_or_failed} non-success rows; successful IDs will be skipped", file=sys.stderr)
    for position, row in enumerate(rows, 1):
        if str(row["qa_id"]) in completed:
            continue
        started = time.monotonic()
        try:
            raw = runner(row, render_prompt(row, config), config["generation"])
            prediction = parse_prediction(raw, row["dataset"], config)
            record = {
                "qa_id": str(row["qa_id"]),
                "source_id": str(row["source_id"]),
                "dataset": row["dataset"],
                "model": args.model,
                "raw_output": raw,
                "normalized_prediction": prediction,
                "terminal_status": "success",
                "elapsed_seconds": time.monotonic() - started,
            }
        except Exception as exc:
            record = {
                "qa_id": str(row["qa_id"]),
                "source_id": str(row["source_id"]),
                "dataset": row["dataset"],
                "model": args.model,
                "raw_output": None,
                "normalized_prediction": "__invalid__",
                "terminal_status": "failure",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
            }
        append_jsonl(args.output, record)
        print(
            f"[{position}/{len(rows)}] {row['qa_id']} {record['terminal_status']} "
            f"{record['normalized_prediction']}",
            flush=True,
        )
        gc.collect()
        try:
            runner.torch.cuda.empty_cache()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--model", required=True, choices=("qwen_omni_3b", "qwen_omni_7b", "minicpm_o_45_gptq"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
