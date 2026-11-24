"""Configuration loading and unresolved-field guards."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _null_paths(value: Any, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        paths = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_null_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_null_paths(child, f"{prefix}[{index}]"))
        return paths
    return []


def load_config(path: str | Path, *, require_resolved: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    if config.get("method") != "sde_cctdh":
        raise ValueError("configuration is not for sde_cctdh")
    for stage, rank in (("sde", 8), ("cctdh", 4)):
        if stage not in config:
            raise ValueError(f"missing {stage} configuration")
        lora_key = "lora" if stage == "sde" else "residual_lora"
        actual = config[stage].get(lora_key, {}).get("rank")
        if int(actual or -1) != rank:
            raise ValueError(f"{stage} LoRA rank must be {rank}, got {actual}")
    if require_resolved:
        unresolved = _null_paths(config)
        if unresolved:
            raise ValueError(f"unresolved configuration fields: {unresolved}")
    return config


def validate_source_disjoint_splits(
    manifest: pd.DataFrame,
    *,
    split_column: str = "formal_split",
    source_column: str = "source_id",
    required_splits: tuple[str, ...] = ("adapt", "tune", "holdout"),
) -> dict[str, int]:
    """Validate source-grouped Adapt/Tune/Holdout boundaries."""
    required = {"qa_id", split_column, source_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"split manifest missing columns: {sorted(missing)}")
    if manifest["qa_id"].duplicated().any():
        raise ValueError("duplicate qa_id in split manifest")
    observed = set(manifest[split_column].astype(str))
    if not set(required_splits).issubset(observed):
        absent = sorted(set(required_splits) - observed)
        raise ValueError(f"missing required splits: {absent}")
    source_split_count = manifest.groupby(source_column)[split_column].nunique()
    leaking = source_split_count[source_split_count.gt(1)]
    if len(leaking):
        raise ValueError(
            f"{len(leaking)} source IDs cross split boundaries: "
            f"{list(map(str, leaking.index[:5]))}"
        )
    return {
        split: int(manifest[manifest[split_column].eq(split)]["qa_id"].nunique())
        for split in required_splits
    }
