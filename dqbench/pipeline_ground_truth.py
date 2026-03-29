"""Load and query Pipeline ground truth manifests."""
from __future__ import annotations
import json
from pathlib import Path

import polars as pl
from pydantic import BaseModel


class PipelineGroundTruth(BaseModel):
    tier: int
    version: str
    rows: int                              # rows in messy input
    planted_issues: int                    # quality issues planted
    duplicate_pairs: list[tuple[int, int]]  # true match pairs
    expected_output_rows: int              # rows after deduplication


def load_pipeline_ground_truth(path: Path) -> PipelineGroundTruth:
    with open(path) as f:
        return PipelineGroundTruth(**json.load(f))


def load_pipeline_clean_df(tier_dir: Path) -> pl.DataFrame:
    return pl.read_csv(tier_dir / "data_clean_deduped.csv", infer_schema_length=0)
