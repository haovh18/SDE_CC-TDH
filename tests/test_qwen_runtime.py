from pathlib import Path

import torch

from sde_cctdh.qwen import (
    LoRALinear,
    QwenProtocol,
    ResidualLoRALinear,
    adapter_state,
    load_adapter_state,
)


ROOT = Path(__file__).resolve().parents[1]


def test_music_protocol_renders_and_normalizes_full_answers():
    protocol = QwenProtocol.from_yaml(
        ROOT / "baseline/config.yaml",
        "music_avqa_42",
    )
    prompt = protocol.render_prompt({"question": "Which instrument?"})
    assert "accordion" in prompt
    assert "accordion, acoustic_guitar" in prompt
    assert "Which instrument?" in prompt
    assert protocol.normalize_answer(" Electric Bass ") == "electric_bass"
    assert protocol.normalize_answer("not-in-vocabulary") == "__invalid__"


def test_residual_lora_stacks_on_frozen_sde_parent():
    torch.manual_seed(18)
    base = torch.nn.Linear(4, 3, bias=False)
    sde = LoRALinear(base, rank=2, alpha=4, dropout=0)
    with torch.no_grad():
        sde.lora_B.fill_(0.1)
    value = torch.randn(2, 4)
    parent_output = sde(value).detach()
    residual = ResidualLoRALinear(sde, rank=1, alpha=2, dropout=0)
    assert torch.allclose(residual(value), parent_output)
    assert not any(parameter.requires_grad for parameter in sde.parameters())
    assert residual.lora_A.requires_grad and residual.lora_B.requires_grad


def test_adapter_state_round_trip_is_strict():
    first = {"layer": LoRALinear(torch.nn.Linear(3, 2), rank=1, alpha=2, dropout=0)}
    second = {"layer": LoRALinear(torch.nn.Linear(3, 2), rank=1, alpha=2, dropout=0)}
    state = adapter_state(first)
    load_adapter_state(second, state)
    assert all(
        torch.equal(state[key], adapter_state(second)[key])
        for key in state
    )
