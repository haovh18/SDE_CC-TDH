from pathlib import Path

import pandas as pd
import pytest

from sde_cctdh.config import load_config, validate_source_disjoint_splits


ROOT = Path(__file__).resolve().parents[1]


def test_source_split_overlap_is_rejected():
    manifest = pd.DataFrame([
        {"qa_id": "a", "source_id": "s", "formal_split": "adapt"},
        {"qa_id": "b", "source_id": "t", "formal_split": "tune"},
        {"qa_id": "c", "source_id": "s", "formal_split": "holdout"},
    ])
    with pytest.raises(ValueError, match="cross split"):
        validate_source_disjoint_splits(manifest)


def test_method_configs_are_resolved():
    config = load_config(ROOT / "configs/qwen3b.yaml")
    assert config["sde"]["lora"]["rank"] == 8
    load_config(ROOT / "configs/qwen3b.yaml", require_resolved=True)
    minicpm = load_config(ROOT / "configs/minicpm.yaml", require_resolved=True)
    assert minicpm["intervention"]["window_length"] == 4
    assert minicpm["teacher"]["hidden_layer"] == 34


def test_method_config_seeds_are_unified():
    def seeds(value):
        if isinstance(value, dict):
            return [
                child
                for key, nested in value.items()
                for child in ([nested] if key == "seed" else seeds(nested))
            ]
        if isinstance(value, list):
            return [child for nested in value for child in seeds(nested)]
        return []

    for name in ("qwen3b.yaml", "minicpm.yaml"):
        assert seeds(load_config(ROOT / "configs" / name)) == [18, 18, 18, 18]
