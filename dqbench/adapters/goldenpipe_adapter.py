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
        """Run the Golden Suite pipeline: GoldenFlow transform → GoldenMatch dedupe."""
        try:
            import goldenflow
            import goldenmatch
        except ImportError:
            raise RuntimeError(
                "goldenpipe requires goldenflow and goldenmatch. "
                "Run: pip install goldenpipe[golden-suite]"
            )

        # Step 1: Read and transform with GoldenFlow
        df = pl.read_csv(csv_path)
        result = goldenflow.transform_df(df)
        cleaned = result.df

        # Step 2: Deduplicate with GoldenMatch
        deduped = goldenmatch.dedupe_df(cleaned)

        return deduped.golden
