"""Source-held-out group reliability and deterministic expert routing."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pandas as pd


def assign_source_folds(
    source_ids: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("at least two source folds are required")
    sources = sorted(set(map(str, source_ids)))
    if len(sources) < n_folds:
        raise ValueError("fewer sources than folds")
    ordered = sorted(
        sources,
        key=lambda source: hashlib.sha256(f"{seed}|{source}".encode()).hexdigest(),
    )
    return {source: index % n_folds for index, source in enumerate(ordered)}


def crossfit_group_reliability(
    expert_audit: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 18,
) -> pd.DataFrame:
    """Compute (correct + 1) / (available + 2) outside each source fold."""
    required = {
        "qa_id",
        "source_id",
        "expert_name",
        "question_group",
        "expert_available",
        "expert_correct",
    }
    missing = required - set(expert_audit.columns)
    if missing:
        raise ValueError(f"expert audit missing columns: {sorted(missing)}")
    rows = expert_audit.copy()
    fold_map = assign_source_folds(
        rows["source_id"].astype(str).tolist(),
        n_folds=n_folds,
        seed=seed,
    )
    rows["source_fold"] = rows["source_id"].astype(str).map(fold_map)
    rows["group_reliability_oof"] = 0.5
    rows["group_reliability_oof_n"] = 0
    for fold in range(n_folds):
        fit = rows[
            rows["source_fold"].ne(fold) & rows["expert_available"].astype(bool)
        ]
        grouped = fit.groupby(["expert_name", "question_group"], dropna=False)[
            "expert_correct"
        ].agg(["sum", "count"])
        index = rows["source_fold"].eq(fold)
        for row_index in rows.index[index]:
            key = (
                rows.at[row_index, "expert_name"],
                rows.at[row_index, "question_group"],
            )
            if key in grouped.index:
                correct = int(grouped.loc[key, "sum"])
                count = int(grouped.loc[key, "count"])
            else:
                correct = count = 0
            rows.at[row_index, "group_reliability_oof"] = (correct + 1) / (
                count + 2
            )
            rows.at[row_index, "group_reliability_oof_n"] = count
    return rows


def route_validated_experts(
    validation_pairs: pd.DataFrame,
    reliability_rows: pd.DataFrame,
    *,
    tie_order: Sequence[str],
) -> pd.DataFrame:
    """Apply Eq. 7 support gates, then Eq. 7/8 reliability routing."""
    required_validation = {
        "qa_id",
        "source_id",
        "question_group",
        "expert_name",
        "expert_support",
        "kappa",
    }
    required_reliability = {
        "qa_id",
        "expert_name",
        "group_reliability_oof",
        "source_fold",
    }
    if required_validation - set(validation_pairs.columns):
        raise ValueError("validation-pair table is incomplete")
    if required_reliability - set(reliability_rows.columns):
        raise ValueError("reliability table is incomplete")
    eligible = validation_pairs[
        validation_pairs["expert_support"].astype(bool)
        & validation_pairs["kappa"].astype(bool)
    ].merge(
        reliability_rows[list(required_reliability)],
        on=["qa_id", "expert_name"],
        how="left",
        validate="many_to_one",
    )
    if eligible["group_reliability_oof"].isna().any():
        raise ValueError("missing source-held-out reliability for an eligible pair")
    tie_rank = {str(expert): index for index, expert in enumerate(tie_order)}
    if not set(eligible["expert_name"].astype(str)).issubset(tie_rank):
        raise ValueError("tie order does not cover every eligible expert")
    eligible["_tie_rank"] = eligible["expert_name"].astype(str).map(tie_rank)
    eligible = eligible.sort_values(
        ["qa_id", "group_reliability_oof", "_tie_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = eligible.groupby("qa_id", sort=False).head(1).copy()
    selected = selected.rename(columns={"expert_name": "selected_expert"})
    selected["kappa"] = True
    return selected.drop(columns=["_tie_rank"]).reset_index(drop=True)
