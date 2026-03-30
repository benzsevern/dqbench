"""Example: Run the DQBench Pipeline benchmark with a custom adapter.

Usage:
    pip install dqbench
    python examples/run_pipeline_benchmark.py
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
from dqbench.adapters.base import PipelineAdapter
from dqbench.runner import run_pipeline_benchmark
from dqbench.report import report_pipeline_rich


class SimplePipelineAdapter(PipelineAdapter):
    """Baseline pipeline adapter -- just lowercases text and deduplicates by email."""

    @property
    def name(self) -> str:
        return "simple-pipeline"

    @property
    def version(self) -> str:
        return "1.0.0"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        df = pl.read_csv(csv_path)

        # Step 1: Basic cleaning
        for col in df.columns:
            if col.startswith("_"):
                continue
            df = df.with_columns(pl.col(col).str.strip_chars().alias(col))

        # Step 2: Deduplicate by email
        if "email" in df.columns:
            df = df.with_columns(
                pl.col("email").str.to_lowercase().alias("_email_key")
            )
            df = df.unique(subset=["_email_key"], keep="first")
            df = df.drop("_email_key")

        return df


if __name__ == "__main__":
    adapter = SimplePipelineAdapter()
    scorecard = run_pipeline_benchmark(adapter)
    report_pipeline_rich(scorecard)
    print(f"\nDQBench Pipeline Score: {scorecard.dqbench_pipeline_score:.2f} / 100")
