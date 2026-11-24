"""Label-audited pair-level and example-level RPU construction.

Reliability and confidence never participate in this module. They belong only
to the second-stage multi-expert router.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

import pandas as pd


class PairState(str, Enum):
    REPAIR = "REPAIR"
    PROTECT = "PROTECT"
    UNCERTAIN = "UNCERTAIN"

    def __str__(self) -> str:
        return self.value


class ExampleState(str, Enum):
    REPAIR = "REPAIR"
    PROTECT = "PROTECT"
    UNCERTAIN = "UNCERTAIN"
    NO_ACTION = "NO_ACTION"

    def __str__(self) -> str:
        return self.value


REQUIRED_COLUMNS = {
    "qa_id",
    "source_id",
    "base_answer",
    "gold_answer",
    "expert_name",
    "expert_answer",
    "expert_available",
}


def _check_input(rows: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"missing RPU columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("RPU input is empty")
    for qa_id, group in rows.groupby("qa_id", sort=False):
        for column in ("source_id", "base_answer", "gold_answer"):
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(f"{qa_id}: inconsistent {column} across experts")
        if group["expert_name"].duplicated().any():
            raise ValueError(f"{qa_id}: duplicate expert pair")


def build_pair_states(
    rows: pd.DataFrame,
    *,
    unavailable_policy: Literal["skip", "uncertain"] = "skip",
) -> pd.DataFrame:
    """Apply Eq. 2 to every expert-example pair.

    With the default ``skip`` policy, unavailable pairs are retained for audit
    but have no pair state. Example-level Uncertain is still derived by Eq. 3.
    """
    _check_input(rows)
    if unavailable_policy not in {"skip", "uncertain"}:
        raise ValueError(f"unknown unavailable policy: {unavailable_policy}")

    out = rows.copy()
    out["base_correct"] = out["base_answer"].eq(out["gold_answer"])
    out["expert_correct"] = out["expert_answer"].eq(out["gold_answer"])
    out["expert_support"] = (
        out["expert_available"].astype(bool) & out["expert_correct"]
    )
    states: list[str | None] = []
    for row in out.itertuples(index=False):
        available = bool(row.expert_available)
        base_correct = bool(row.base_correct)
        expert_correct = bool(row.expert_correct)
        if not available and unavailable_policy == "skip":
            states.append(None)
        elif not base_correct and available and expert_correct:
            states.append(PairState.REPAIR.value)
        elif base_correct and available and not expert_correct:
            states.append(PairState.PROTECT.value)
        elif not base_correct:
            states.append(PairState.UNCERTAIN.value)
        else:
            states.append(None)
    out["pair_state"] = pd.array(states, dtype="string")
    return out


def aggregate_example_states(pair_rows: pd.DataFrame) -> pd.DataFrame:
    """Apply Eq. 3 without treating No Action as a fourth RPU obligation."""
    required = {"qa_id", "source_id", "base_answer", "gold_answer", "pair_state"}
    missing = required - set(pair_rows.columns)
    if missing:
        raise ValueError(f"missing pair-audit columns: {sorted(missing)}")

    records = []
    for qa_id, group in pair_rows.groupby("qa_id", sort=False):
        first = group.iloc[0]
        base_correct = str(first["base_answer"]) == str(first["gold_answer"])
        pair_states = set(group["pair_state"].dropna().astype(str))
        if not base_correct and PairState.REPAIR.value in pair_states:
            state = ExampleState.REPAIR
        elif base_correct and PairState.PROTECT.value in pair_states:
            state = ExampleState.PROTECT
        elif not base_correct:
            state = ExampleState.UNCERTAIN
        else:
            state = ExampleState.NO_ACTION
        records.append(
            {
                "qa_id": qa_id,
                "source_id": first["source_id"],
                "base_answer": first["base_answer"],
                "gold_answer": first["gold_answer"],
                "base_correct": base_correct,
                "example_state": state.value,
                "repair_experts": sorted(
                    group.loc[
                        group["pair_state"].eq(PairState.REPAIR.value),
                        "expert_name",
                    ].astype(str)
                ),
                "protect_experts": sorted(
                    group.loc[
                        group["pair_state"].eq(PairState.PROTECT.value),
                        "expert_name",
                    ].astype(str)
                ),
            }
        )
        for column in (
            "dataset",
            "question_group",
            "media_path",
            "question",
            "question_text",
            "options",
            "multi_choice",
        ):
            if column in group.columns:
                values = group[column].tolist()
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(f"{qa_id}: inconsistent {column} across experts")
                records[-1][column] = values[0]
    return pd.DataFrame.from_records(records)


def build_rpu_manifest(
    rows: pd.DataFrame,
    *,
    unavailable_policy: Literal["skip", "uncertain"] = "skip",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the pair audit and the mutually exclusive example manifest."""
    pairs = build_pair_states(rows, unavailable_policy=unavailable_policy)
    examples = aggregate_example_states(pairs)
    return pairs, examples
