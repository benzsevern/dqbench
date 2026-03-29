"""Tests for Pipeline ground truth model and loaders."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import polars as pl

from dqbench.pipeline_ground_truth import (
    PipelineGroundTruth,
    load_pipeline_ground_truth,
    load_pipeline_clean_df,
)


def test_pipeline_ground_truth_creation():
    gt = PipelineGroundTruth(
        tier=1,
        version="1.0.0",
        rows=1000,
        planted_issues=150,
        duplicate_pairs=[(0, 900), (1, 901)],
        expected_output_rows=900,
    )
    assert gt.tier == 1
    assert gt.planted_issues == 150
    assert gt.expected_output_rows == 900


def test_pipeline_ground_truth_roundtrip():
    gt = PipelineGroundTruth(
        tier=1,
        version="1.0.0",
        rows=1000,
        planted_issues=100,
        duplicate_pairs=[(5, 950)],
        expected_output_rows=999,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt.model_dump(), f)
        path = Path(f.name)

    loaded = load_pipeline_ground_truth(path)
    assert loaded.tier == gt.tier
    assert loaded.rows == gt.rows
    assert loaded.duplicate_pairs == gt.duplicate_pairs
    path.unlink()


def test_load_pipeline_clean_df():
    with tempfile.TemporaryDirectory() as tmpdir:
        tier_dir = Path(tmpdir)
        df = pl.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
        df.write_csv(tier_dir / "data_clean_deduped.csv")

        loaded = load_pipeline_clean_df(tier_dir)
        assert loaded.shape == (2, 2)
        assert loaded.columns == ["a", "b"]
