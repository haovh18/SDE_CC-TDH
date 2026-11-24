import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sde_cctdh.config import load_config
from sde_cctdh.minicpm import (
    MiniCPMRuntime,
    ResidualGPTQLoRALinear,
    adapter_state,
    create_runtime,
    install_residual_lora,
    install_sde_lora,
    response_token_nll,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeQuantLinear(torch.nn.Module):
    """Uses AutoGPTQ's compact dimension attribute spelling."""

    def __init__(self, in_features=8, out_features=8):
        super().__init__()
        self.infeatures = in_features
        self.outfeatures = out_features
        self.register_buffer("qweight", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("dense_weight", torch.randn(out_features, in_features))

    def forward(self, value):
        return torch.nn.functional.linear(value, self.dense_weight)


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, FakeQuantLinear())


class FakeBlock(torch.nn.Module):
    def __init__(self, offset=0.0):
        super().__init__()
        self.self_attn = FakeAttention()
        self.offset = offset

    def forward(self, hidden):
        return hidden + self.offset


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeBlock(float(i)) for i in range(36)])


class FakeLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()


class FakeMiniCPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = FakeLLM()

    def forward(self, data, **_kwargs):
        hidden = data["hidden"]
        for layer in self.llm.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=torch.empty(0))


def test_gptq_targets_are_exactly_layers_28_through_35_qkvo():
    model = FakeMiniCPM()
    modules = install_sde_lora(
        model,
        layers=range(28, 36),
        rank=8,
        alpha=16,
        dropout=0,
        dtype=torch.float32,
    )
    assert len(modules) == 8 * 4
    assert set(name.rsplit(".", 1)[-1] for name in modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert {int(name.split(".")[3]) for name in modules} == set(range(28, 36))


def test_response_nll_ignores_prompt_and_normalizes_each_response_length():
    logits = torch.zeros(2, 5, 3)
    labels = torch.full((2, 5), -100)
    labels[0, 2] = 0
    labels[1, 2:4] = torch.tensor([0, 1])
    logits[0, 1, 0] = 2.0
    logits[1, 1, 0] = 2.0
    logits[1, 2, 1] = 1.0
    actual = response_token_nll(logits, labels)
    first = torch.nn.functional.cross_entropy(logits[0, 1:2], labels[0, 2:3])
    second = (
        torch.nn.functional.cross_entropy(logits[1, 1:2], labels[1, 2:3])
        + torch.nn.functional.cross_entropy(logits[1, 2:3], labels[1, 3:4])
    ) / 2
    assert torch.allclose(actual, torch.stack([first, second]))
    changed_prompt = logits.clone()
    changed_prompt[:, 0] = 1000
    assert torch.allclose(response_token_nll(changed_prompt, labels), actual)


def test_frozen_sde_parent_and_residual_are_the_only_trainable_parameters():
    model = FakeMiniCPM()
    sde = install_sde_lora(
        model,
        layers=range(28, 36),
        rank=8,
        alpha=16,
        dropout=0,
        dtype=torch.float32,
    )
    residual = install_residual_lora(
        model,
        layers=range(28, 36),
        rank=4,
        alpha=8,
        dropout=0,
        dtype=torch.float32,
    )
    assert all(isinstance(module, ResidualGPTQLoRALinear) for module in residual.values())
    assert not any(parameter.requires_grad for module in sde.values() for parameter in module.parameters())
    expected = {
        id(parameter)
        for module in residual.values()
        for parameter in (module.lora_A, module.lora_B)
    }
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert actual == expected


def test_answer_position_hidden_extraction_uses_response_start_minus_one():
    runtime = MiniCPMRuntime.__new__(MiniCPMRuntime)
    runtime.model = FakeMiniCPM()
    runtime.prepare_training = lambda _row, _answer: (
        {"hidden": torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)},
        {"response_start": 3},
    )

    records = runtime.extract_hidden(
        [{"qa_id": "q", "source_id": "s", "gold_response": "A"}],
        layer=34,
    )
    expected = (
        torch.arange(40).reshape(1, 5, 8)[0, 2] + sum(range(35))
    ).tolist()
    assert records[0]["parent_hidden"] == expected
    assert len(records[0]["parent_hidden"]) == 8
    assert records[0]["answer_position"] == "response_start_minus_one"


def test_adapter_state_contains_only_low_rank_weights():
    model = FakeMiniCPM()
    modules = install_sde_lora(
        model,
        layers=[28],
        rank=8,
        alpha=16,
        dropout=0,
        dtype=torch.float32,
    )
    state = adapter_state(modules)
    assert len(state) == 8
    assert all(key.endswith(("lora_A", "lora_B")) for key in state)
    assert not any("qweight" in key or "dense_weight" in key for key in state)


def test_exported_adapters_load_in_order_and_stack_for_decode(tmp_path):
    config = load_config(ROOT / "configs/minicpm.yaml", require_resolved=True)
    source = MiniCPMRuntime.__new__(MiniCPMRuntime)
    source.model = FakeMiniCPM()
    source.adapter_dtype = torch.float32
    source.config = config
    source.model_path = Path("/models/MiniCPM-o-4_5-GPTQ")
    source.model_config_hash = "model-hash"
    source.protocol = SimpleNamespace(name="avqa_4way")
    source.protocol_config_hash = "protocol-hash"
    source.sde_modules = install_sde_lora(
        source.model,
        layers=range(28, 36),
        rank=8,
        alpha=16,
        dropout=0,
        dtype=torch.float32,
    )
    source.modules = install_residual_lora(
        source.model,
        layers=range(28, 36),
        rank=4,
        alpha=8,
        dropout=0,
        dtype=torch.float32,
    )
    source.parent_adapter_hash = "training-parent-hash"
    source.residual_adapter_hash = "training-residual-hash"
    source.save_deployed(tmp_path)

    deployed = MiniCPMRuntime.__new__(MiniCPMRuntime)
    deployed.model = FakeMiniCPM()
    deployed.adapter_dtype = torch.float32
    deployed.config = config
    deployed.model_config_hash = "model-hash"
    deployed.protocol = SimpleNamespace(name="avqa_4way")
    deployed.protocol_config_hash = "protocol-hash"
    deployed.sde_modules = install_sde_lora(
        deployed.model,
        layers=range(28, 36),
        rank=8,
        alpha=16,
        dropout=0,
        dtype=torch.float32,
    )
    deployed.modules = deployed.sde_modules
    deployed.parent_adapter_hash = None
    deployed.residual_adapter_hash = None
    deployed.load_sde_for_inference(tmp_path / "sde_adapter.pt")
    deployed.load_cctdh_for_inference(tmp_path / "cctdh_adapter.pt")
    assert len(deployed.modules) == 32
    assert all(
        isinstance(module, ResidualGPTQLoRALinear)
        for module in deployed.modules.values()
    )
    assert not any(parameter.requires_grad for parameter in deployed.model.parameters())


@pytest.mark.skipif(
    not (Path(os.environ.get("MINICPM_MODEL_PATH", "")) / "config.json").is_file(),
    reason="set MINICPM_MODEL_PATH to run the 11 GB GPTQ load test",
)
def test_real_minicpm_gptq_model_loads_and_installs_lora():
    runtime = create_runtime(
        stage="sde",
        config=load_config(ROOT / "configs/minicpm.yaml", require_resolved=True),
        model_path=Path(os.environ["MINICPM_MODEL_PATH"]),
        protocol_config=ROOT / "baseline/config.yaml",
        device="auto",
        seed=18,
    )
    try:
        assert len(runtime.sde_modules) == 32
        quantization = runtime.model.config.quantization_config
        method = (
            quantization.get("quant_method")
            if isinstance(quantization, dict)
            else getattr(quantization, "quant_method")
        )
        assert str(method).lower().endswith("gptq")
        projection = runtime.sde_modules[
            "llm.model.layers.28.self_attn.q_proj"
        ]
        value = torch.zeros(
            1,
            1,
            projection.lora_A.shape[1],
            device=projection.lora_A.device,
            dtype=projection.lora_A.dtype,
        )
        assert projection(value).shape[-1] == projection.lora_B.shape[0]
    finally:
        runtime.close()
