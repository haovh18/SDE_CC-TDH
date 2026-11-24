"""Paper-defined paired transition and RPU metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rpu import ExampleState


def _ratio(numerator: int, denominator: int) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def evaluate_predictions(rows: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "base_answer",
        "method_answer",
        "gold_answer",
        "example_state",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    base_correct = rows["base_answer"].eq(rows["gold_answer"])
    method_correct = rows["method_answer"].eq(rows["gold_answer"])
    w2c = int((~base_correct & method_correct).sum())
    c2w = int((base_correct & ~method_correct).sum())
    repair = rows["example_state"].eq(ExampleState.REPAIR.value)
    protect = rows["example_state"].eq(ExampleState.PROTECT.value)
    uncertain = rows["example_state"].eq(ExampleState.UNCERTAIN.value)
    return {
        "Accuracy": float(method_correct.mean()),
        "W2C": w2c,
        "C2W": c2w,
        "Net": w2c - c2w,
        "BER": _ratio(w2c, w2c + c2w),
        "DR": _ratio(c2w, int(base_correct.sum())),
        "RR": _ratio(int((repair & method_correct).sum()), int(repair.sum())),
        "PR": _ratio(int((protect & method_correct).sum()), int(protect.sum())),
        "UFR": _ratio(
            int((uncertain & rows["method_answer"].ne(rows["base_answer"])).sum()),
            int(uncertain.sum()),
        ),
    }


def source_bootstrap(
    rows: pd.DataFrame,
    *,
    metric: str,
    repetitions: int,
    seed: int = 18,
) -> dict[str, float]:
    if "source_id" not in rows:
        raise ValueError("source bootstrap requires source_id")
    sources = np.array(sorted(rows["source_id"].astype(str).unique()), dtype=object)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = generator.choice(sources, len(sources), replace=True)
        sample = pd.concat(
            [rows[rows["source_id"].astype(str).eq(source)] for source in sampled],
            ignore_index=True,
        )
        values.append(float(evaluate_predictions(sample)[metric]))
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "mean": float(finite.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }
