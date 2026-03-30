"""GoldenPipe adapter for DQBench Pipeline benchmarks."""
from __future__ import annotations
from pathlib import Path
import polars as pl
from dqbench.adapters.base import PipelineAdapter


class GoldenPipeAdapter(PipelineAdapter):
    @property
    def name(self) -> str:
        return "goldenpipe"

    @property
    def version(self) -> str:
        try:
            import goldenpipe
            return goldenpipe.__version__
        except ImportError:
            return "not-installed"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        """Run the Golden Suite pipeline: transform → deduplicate.

        Uses GoldenFlow and GoldenMatch directly (same stages GoldenPipe
        orchestrates) to get the final DataFrame, since Pipeline.run()
        returns PipeResult without the DataFrame.
        """
        try:
            import goldenflow
            import goldenmatch
        except ImportError:
            raise RuntimeError(
                "goldenpipe[golden-suite] is not installed. "
                "Run: pip install goldenflow goldenmatch"
            )

        # Stage 1: Transform with GoldenFlow (fix quality issues)
        df = pl.read_csv(csv_path)
        transform_result = goldenflow.transform_df(df)
        cleaned = transform_result.df

        # Stage 2: Deduplicate with GoldenMatch (resolve entities)
        dedupe_result = goldenmatch.dedupe_df(
            cleaned,
            fuzzy={
                "first_name": 0.65,
                "last_name": 0.75,
                "email": 0.85,
            },
            exact=["email"],
            threshold=0.60,
        )

        # Return golden records (the deduplicated output)
        if dedupe_result.golden is not None and dedupe_result.golden.shape[0] > 0:
            return dedupe_result.golden
        # If no duplicates found, return cleaned data as-is
        return cleaned
