"""Scoring logic for transform benchmarks."""
from __future__ import annotations

import polars as pl

from dqbench.models import TransformColumnResult, TransformTierResult


def score_transform_tier(
    result_df: pl.DataFrame,
    clean_df: pl.DataFrame,
    messy_df: pl.DataFrame,
    tier: int,
    time_seconds: float,
    memory_mb: float,
) -> TransformTierResult:
    """Score a transform tool's output against the clean ground truth."""
    # Validate shape
    if result_df.shape[0] != clean_df.shape[0]:
        return _zero_result(
            tier, time_seconds, memory_mb,
            f"Row count mismatch: {result_df.shape[0]} vs {clean_df.shape[0]}",
        )

    if set(result_df.columns) != set(clean_df.columns):
        return _zero_result(tier, time_seconds, memory_mb, "Column mismatch")

    per_column: list[TransformColumnResult] = []
    total_planted = 0
    total_correct = 0
    total_wrong = 0
    total_skipped = 0

    for col in clean_df.columns:
        if col not in result_df.columns:
            continue

        clean_col = clean_df[col].cast(pl.Utf8).fill_null("")
        messy_col = messy_df[col].cast(pl.Utf8).fill_null("")
        result_col = result_df[col].cast(pl.Utf8).fill_null("")

        # Only score cells that differ between messy and clean (planted issues)
        planted_mask = clean_col != messy_col
        planted_count = planted_mask.sum()

        if planted_count == 0:
            continue

        # Compare result against clean for planted cells only
        correct_mask = (result_col == clean_col) & planted_mask
        skipped_mask = (result_col == messy_col) & planted_mask

        correct = correct_mask.sum()
        skipped = skipped_mask.sum()
        wrong = planted_count - correct

        per_column.append(TransformColumnResult(
            column=col,
            planted_cells=int(planted_count),
            correct_cells=int(correct),
            wrong_cells=int(wrong),
            skipped_cells=int(skipped),
            accuracy=correct / planted_count if planted_count > 0 else 0.0,
        ))

        total_planted += planted_count
        total_correct += correct
        total_wrong += wrong
        total_skipped += skipped

    accuracy = total_correct / total_planted if total_planted > 0 else 0.0

    return TransformTierResult(
        tier=tier,
        accuracy=float(accuracy),
        correct_cells=int(total_correct),
        wrong_cells=int(total_wrong),
        skipped_cells=int(total_skipped),
        planted_cells=int(total_planted),
        time_seconds=time_seconds,
        memory_mb=memory_mb,
        per_column=per_column,
    )


def _zero_result(
    tier: int,
    time_seconds: float,
    memory_mb: float,
    error: str,
) -> TransformTierResult:
    return TransformTierResult(
        tier=tier,
        accuracy=0.0,
        correct_cells=0,
        wrong_cells=0,
        skipped_cells=0,
        planted_cells=0,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
        per_column=[],
    )
