"""Tests for Pipeline runner integration."""
from __future__ import annotations
from pathlib import Path

import polars as pl

from dqbench.adapters.base import PipelineAdapter
from dqbench.models import PipelineScorecard


class PassthroughPipelineAdapter(PipelineAdapter):
    """Returns the clean+deduped ground truth as the pipeline output."""

    @property
    def name(self) -> str:
        return "passthrough-pipeline"

    @property
    def version(self) -> str:
        return "0.0.1"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        # Load the clean ground truth (cheating for test purposes)
        clean_path = csv_path.parent / "data_clean_deduped.csv"
        return pl.read_csv(clean_path, infer_schema_length=0)


class TestPipelineRunner:
    def test_run_pipeline_benchmark_returns_scorecard(self):
        from dqbench.runner import run_pipeline_benchmark, ensure_pipeline_datasets
        ensure_pipeline_datasets()
        adapter = PassthroughPipelineAdapter()
        scorecard = run_pipeline_benchmark(adapter, tiers=[1])
        assert isinstance(scorecard, PipelineScorecard)
        assert scorecard.tool_name == "passthrough-pipeline"
        assert len(scorecard.tiers) == 1

    def test_perfect_pipeline_scores(self):
        from dqbench.runner import run_pipeline_benchmark, ensure_pipeline_datasets
        ensure_pipeline_datasets()
        adapter = PassthroughPipelineAdapter()
        scorecard = run_pipeline_benchmark(adapter, tiers=[1])
        assert scorecard.tiers[0].transform_accuracy == 1.0
        assert scorecard.tiers[0].dedup_accuracy == 1.0
        assert scorecard.tiers[0].composite == 1.0

    def test_ensure_pipeline_datasets_creates_files(self):
        from dqbench.runner import CACHE_DIR, ensure_pipeline_datasets
        ensure_pipeline_datasets()
        assert (CACHE_DIR / "pipeline_tier1" / "data.csv").exists()
        assert (CACHE_DIR / "pipeline_tier1" / "data_clean_deduped.csv").exists()
        assert (CACHE_DIR / "pipeline_tier1" / "pipeline_ground_truth.json").exists()
