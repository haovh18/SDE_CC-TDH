"""Stage artifacts that connect RPU, SDE parent replay, and CC-TDH."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from .cctdh import (
    build_parent_retention_pools,
    build_validated_repair_pool,
    mine_residual_repairs,
)
from .handoff import fit_ridge_teacher
from .handoff import build_detail_target, fit_detail_transform
from .intervention import validate_effect_table
from .routing import crossfit_group_reliability, route_validated_experts


def write_jsonl(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    records = frame.astype(object).where(pd.notnull(frame), None).to_dict("records")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_response_cache(
    manifest: pd.DataFrame,
    base_outputs: pd.DataFrame,
) -> pd.DataFrame:
    """Join deterministic Base responses to full gold-response targets."""
    required_manifest = {"qa_id", "source_id", "gold_answer"}
    required_base = {"qa_id", "base_answer", "base_response"}
    if required_manifest - set(manifest.columns):
        raise ValueError("manifest lacks response-cache fields")
    if required_base - set(base_outputs.columns):
        raise ValueError("Base outputs lack response-cache fields")
    if manifest["qa_id"].duplicated().any() or base_outputs["qa_id"].duplicated().any():
        raise ValueError("response-cache inputs must be unique by qa_id")
    joined = manifest.merge(
        base_outputs[list(required_base)],
        on="qa_id",
        how="left",
        validate="one_to_one",
    )
    if joined["base_response"].isna().any():
        raise ValueError("Base outputs do not cover the manifest")
    joined["gold_response"] = joined.get("gold_response", joined["gold_answer"])
    return joined.copy()


def build_expert_pair_input(
    base_outputs: pd.DataFrame,
    expert_predictions: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the long-form, label-audited input consumed by the RPU builder."""
    required_base = {"qa_id", "source_id", "gold_answer", "base_answer"}
    required_expert = {
        "qa_id",
        "expert_name",
        "expert_answer",
        "expert_available",
    }
    if required_base - set(base_outputs.columns):
        raise ValueError("Base output table is incomplete")
    if required_expert - set(expert_predictions.columns):
        raise ValueError("expert prediction table is incomplete")
    base = base_outputs.copy()
    if metadata is not None:
        columns = [
            column
            for column in ("qa_id", "question_group", "media_path", "question", "options")
            if column in metadata
        ]
        base = base.merge(
            metadata[columns],
            on="qa_id",
            how="left",
            validate="one_to_one",
        )
    expert_predictions = expert_predictions[
        expert_predictions["qa_id"].astype(str).isin(
            set(base["qa_id"].astype(str))
        )
    ].copy()
    if expert_predictions.empty:
        raise ValueError("expert predictions do not cover any Base QA")
    duplicate = expert_predictions.duplicated(["qa_id", "expert_name"])
    if duplicate.any():
        raise ValueError("expert predictions repeat a QA/expert pair")
    pairs = expert_predictions.merge(
        base,
        on="qa_id",
        how="left",
        validate="many_to_one",
        suffixes=("_expert", ""),
    )
    if pairs["base_answer"].isna().any():
        raise ValueError("expert predictions contain QA absent from Base outputs")
    for column in ("source_id", "gold_answer"):
        expert_column = f"{column}_expert"
        if expert_column in pairs:
            mismatch = pairs[expert_column].astype(str).ne(pairs[column].astype(str))
            if mismatch.any():
                raise ValueError(f"expert and Base tables disagree on {column}")
            pairs = pairs.drop(columns=expert_column)
    if "question_group_expert" in pairs:
        if "question_group" not in pairs:
            pairs = pairs.rename(
                columns={"question_group_expert": "question_group"}
            )
        else:
            pairs["question_group"] = pairs["question_group"].fillna(
                pairs["question_group_expert"]
            )
            pairs = pairs.drop(columns="question_group_expert")
    return pairs


def _attach_raw_details(
    routes: pd.DataFrame,
    local_details: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "qa_id",
        "expert_name",
        "target_pooled_feature",
        "control_pooled_feature",
    }
    if required - set(local_details.columns):
        raise ValueError("local-detail table is incomplete")
    details = local_details.rename(columns={"expert_name": "selected_expert"})
    columns = [
        "qa_id",
        "selected_expert",
        "target_pooled_feature",
        "control_pooled_feature",
    ]
    joined = routes.merge(
        details[columns],
        on=["qa_id", "selected_expert"],
        how="left",
        validate="one_to_one",
    )
    if joined["target_pooled_feature"].isna().any():
        raise ValueError("a routed pair has no local detail features")
    return joined


def _build_disjoint_detail_sets(
    *,
    all_routes: pd.DataFrame,
    validated_repair: pd.DataFrame,
    local_details: pd.DataFrame,
    pair_audit: pd.DataFrame,
    latent_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, torch.Tensor]]]:
    """Fit Q_e outside residual-supervision sources and transform both sets."""
    supervision = _attach_raw_details(validated_repair, local_details)
    supervision_sources = set(supervision["source_id"].astype(str))
    fit_routes = all_routes[
        ~all_routes["source_id"].astype(str).isin(supervision_sources)
    ].copy()
    fit_routes = _attach_raw_details(fit_routes, local_details)
    audit_columns = [
        column
        for column in pair_audit.columns
        if column not in fit_routes.columns or column in {"qa_id", "expert_name"}
    ]
    fit_routes = fit_routes.merge(
        pair_audit[audit_columns].rename(
            columns={"expert_name": "selected_expert"}
        ),
        on=["qa_id", "selected_expert"],
        how="left",
        validate="one_to_one",
    )

    transforms: dict[str, dict[str, torch.Tensor]] = {}
    fit_parts = []
    supervision_parts = []
    required_experts = set(supervision["selected_expert"].astype(str))
    for expert in sorted(required_experts):
        fit_rows = fit_routes[fit_routes["selected_expert"].astype(str).eq(expert)].copy()
        use_rows = supervision[
            supervision["selected_expert"].astype(str).eq(expert)
        ].copy()
        if len(fit_rows) < 2:
            raise ValueError(
                f"{expert}: fewer than two source-disjoint teacher-fit details"
            )
        raw_fit = torch.stack(
            [
                torch.as_tensor(target, dtype=torch.float32)
                - torch.as_tensor(control, dtype=torch.float32)
                for target, control in zip(
                    fit_rows["target_pooled_feature"],
                    fit_rows["control_pooled_feature"],
                    strict=True,
                )
            ]
        )
        transform = fit_detail_transform(
            expert,
            raw_fit,
            latent_dim=latent_dim,
        )
        transforms[expert] = {
            "mean": transform.mean.cpu(),
            "components": transform.components.cpu(),
            "std": transform.std.cpu(),
            "eps": torch.tensor(transform.eps),
            "fit_sources": sorted(set(fit_rows["source_id"].astype(str))),
        }
        for rows, destination in (
            (fit_rows, fit_parts),
            (use_rows, supervision_parts),
        ):
            rows = rows.copy()
            rows["detail_target"] = [
                build_detail_target(
                    torch.as_tensor(target, dtype=torch.float32),
                    torch.as_tensor(control, dtype=torch.float32),
                    transform,
                ).cpu().tolist()
                for target, control in zip(
                    rows["target_pooled_feature"],
                    rows["control_pooled_feature"],
                    strict=True,
                )
            ]
            destination.append(rows)
    fit = pd.concat(fit_parts, ignore_index=True)
    use = pd.concat(supervision_parts, ignore_index=True)
    fit = fit.rename(columns={"selected_expert": "expert_name"})
    return fit, use, transforms


def prepare_cctdh_artifacts(
    *,
    sde_repair_pool: pd.DataFrame,
    example_manifest: pd.DataFrame,
    pair_audit: pd.DataFrame,
    parent_outputs: pd.DataFrame,
    intervention_effects: pd.DataFrame,
    detail_targets: pd.DataFrame | None,
    n_folds: int,
    seed: int,
    tie_order: Sequence[str],
    local_details: pd.DataFrame | None = None,
    latent_dim: int = 8,
) -> dict[str, Any]:
    """Build Eq. 6-Eq. 9 pools from immutable first-stage and expert artifacts."""
    residual = mine_residual_repairs(sde_repair_pool, parent_outputs, pair_audit)
    protect, anchor = build_parent_retention_pools(example_manifest, parent_outputs)

    validation = validate_effect_table(intervention_effects)
    pair_fields = pair_audit[
        [
            column
            for column in (
                "qa_id",
                "source_id",
                "question_group",
                "expert_name",
                "expert_support",
                "expert_available",
                "expert_correct",
            )
            if column in pair_audit
        ]
    ].copy()
    merge_keys = ["qa_id", "expert_name"]
    for column in ("source_id", "question_group", "expert_support"):
        if column not in validation:
            validation = validation.merge(
                pair_fields[merge_keys + [column]],
                on=merge_keys,
                how="left",
                validate="many_to_one",
            )
    if validation[["source_id", "question_group", "expert_support"]].isna().any().any():
        raise ValueError("intervention rows do not align with the pair audit")

    reliability = crossfit_group_reliability(
        pair_audit,
        n_folds=n_folds,
        seed=seed,
    )
    routed = route_validated_experts(
        validation,
        reliability,
        tie_order=tie_order,
    )
    validated = build_validated_repair_pool(residual, routed)
    teacher_fit = None
    transforms = None
    if local_details is not None:
        teacher_fit, validated, transforms = _build_disjoint_detail_sets(
            all_routes=routed,
            validated_repair=validated,
            local_details=local_details,
            pair_audit=pair_audit,
            latent_dim=latent_dim,
        )
    else:
        if detail_targets is None:
            raise ValueError("detail targets or raw local details are required")
        required_detail = {"qa_id", "expert_name", "detail_target"}
        if required_detail - set(detail_targets.columns):
            raise ValueError("detail-target table is incomplete")
        details = detail_targets.rename(columns={"expert_name": "selected_expert"})
        validated = validated.merge(
            details[["qa_id", "selected_expert", "detail_target"]],
            on=["qa_id", "selected_expert"],
            how="left",
            validate="one_to_one",
        )
        if validated["detail_target"].isna().any():
            raise ValueError("a routed Repair example has no detail target")
    if "gold_response" not in validated:
        validated["gold_response"] = validated["gold_answer"]
    result = {
        "residual_repair": residual,
        "parent_protect": protect,
        "parent_anchor": anchor,
        "intervention_validation": validation,
        "reliability": reliability,
        "routes": routed,
        "validated_repair": validated,
    }
    if teacher_fit is not None:
        result["teacher_fit"] = teacher_fit
        result["detail_transforms"] = transforms
    return result


def fit_teacher_bundle(
    teacher_fit: pd.DataFrame,
    supervision: pd.DataFrame,
    *,
    hidden_layer: int,
    ridge: float,
) -> dict[str, Any]:
    """Fit one source-disjoint Eq. 8 map per selected expert."""
    required_fit = {
        "source_id",
        "expert_name",
        "parent_hidden",
        "detail_target",
    }
    required_use = {"source_id", "selected_expert"}
    if required_fit - set(teacher_fit.columns):
        raise ValueError("teacher-fit table is incomplete")
    if required_use - set(supervision.columns):
        raise ValueError("teacher supervision table is incomplete")
    teachers: dict[str, dict[str, Any]] = {}
    for expert, rows in teacher_fit.groupby("expert_name", sort=True):
        use = supervision[supervision["selected_expert"].eq(expert)]
        if use.empty:
            continue
        hidden = torch.as_tensor(
            rows["parent_hidden"].tolist(),
            dtype=torch.float32,
        )
        targets = torch.as_tensor(
            rows["detail_target"].tolist(),
            dtype=torch.float32,
        )
        teacher = fit_ridge_teacher(
            str(expert),
            hidden,
            targets,
            fit_source_ids=rows["source_id"].astype(str).tolist(),
            supervision_source_ids=use["source_id"].astype(str).tolist(),
            ridge=ridge,
        )
        teachers[str(expert)] = {
            "weight": teacher.weight.cpu(),
            "ridge": teacher.ridge,
            "fit_sources": sorted(teacher.fit_sources),
        }
    required_experts = set(supervision["selected_expert"].astype(str))
    if set(teachers) != required_experts:
        missing = sorted(required_experts - set(teachers))
        raise ValueError(f"teacher-fit table has no map for routes: {missing}")
    return {
        "hidden_layer": int(hidden_layer),
        "answer_position": "response_start_minus_one",
        "teachers": teachers,
    }
