"""Scoring logic for Pipeline benchmarks."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from dqbench.models import PipelineTierResult
from dqbench.pipeline_ground_truth import load_pipeline_clean_df


def score_pipeline_tier(
    result_df: pl.DataFrame,
    tier_dir: Path,
    tier: int,
    time_seconds: float,
    memory_mb: float,
    expected_rows: int,
) -> PipelineTierResult:
    """Score a pipeline tool's output against the clean+deduplicated ground truth."""
    clean_df = load_pipeline_clean_df(tier_dir)
    messy_df = pl.read_csv(tier_dir / "data.csv", infer_schema_length=0)

    output_rows = result_df.shape[0]

    # ---- Dedup accuracy ----
    if expected_rows == 0:
        dedup_accuracy = 0.0
    else:
        dedup_accuracy = max(0.0, 1.0 - abs(output_rows - expected_rows) / expected_rows)

    # ---- Transform accuracy ----
    # Join result with clean ground truth on _row_id
    if output_rows == 0 or "_row_id" not in result_df.columns:
        transform_accuracy = 0.0
    else:
        # Cast all to string for comparison
        result_str = result_df.cast({col: pl.Utf8 for col in result_df.columns})
        clean_str = clean_df.cast({col: pl.Utf8 for col in clean_df.columns})
        messy_str = messy_df.cast({col: pl.Utf8 for col in messy_df.columns})

        # Join on _row_id
        joined = result_str.join(
            clean_str, on="_row_id", suffix="_clean", how="inner"
        )
        # Also join messy to know which cells were planted
        joined = joined.join(
            messy_str.select([
                pl.col("_row_id"),
                *[pl.col(c).alias(f"{c}_messy") for c in messy_str.columns if c != "_row_id"],
            ]),
            on="_row_id", how="left",
        )

        # Score columns (exclude _row_id)
        score_cols = [c for c in clean_df.columns if c != "_row_id"]
        total_planted = 0
        total_correct = 0

        for col in score_cols:
            if col not in result_str.columns:
                continue
            clean_col = joined[f"{col}_clean"].fill_null("")
            messy_col_name = f"{col}_messy"
            if messy_col_name in joined.columns:
                messy_col = joined[messy_col_name].fill_null("")
            else:
                continue
            result_col = joined[col].fill_null("")

            # Only count cells that differ between messy and clean (planted issues)
            planted_mask = clean_col != messy_col
            planted_count = planted_mask.sum()
            if planted_count == 0:
                continue

            correct_mask = (result_col == clean_col) & planted_mask
            correct = correct_mask.sum()

            total_planted += int(planted_count)
            total_correct += int(correct)

        transform_accuracy = total_correct / total_planted if total_planted > 0 else 0.0

    composite = transform_accuracy * 0.6 + dedup_accuracy * 0.4

    return PipelineTierResult(
        tier=tier,
        transform_accuracy=float(transform_accuracy),
        dedup_accuracy=float(dedup_accuracy),
        composite=float(composite),
        output_rows=output_rows,
        expected_rows=expected_rows,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
    )
