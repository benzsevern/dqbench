"""GoldenPipe adapter for DQBench Pipeline benchmarks."""
from __future__ import annotations
from pathlib import Path
import polars as pl
from dqbench.adapters.base import PipelineAdapter


class GoldenPipeAdapter(PipelineAdapter):
    @property
    def name(self) -> str: return "goldenpipe"

    @property
    def version(self) -> str:
        try:
            import goldenpipe
            return goldenpipe.__version__
        except ImportError:
            return "not-installed"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        try:
            from goldenpipe import Pipeline
        except ImportError:
            raise RuntimeError("goldenpipe is not installed. Run: pip install goldenpipe[golden-suite]")
        pipeline = Pipeline()
        result = pipeline.run(csv_path)
        return result.df
