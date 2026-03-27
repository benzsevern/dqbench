from __future__ import annotations

from pathlib import Path

import polars as pl

from dqbench.adapters.base import TransformAdapter


class GoldenFlowAdapter(TransformAdapter):
    @property
    def name(self) -> str:
        return "GoldenFlow"

    @property
    def version(self) -> str:
        try:
            from goldenflow import __version__
            return __version__
        except ImportError:
            return "not installed"

    def transform(self, csv_path: Path) -> pl.DataFrame:
        from goldenflow.config.schema import GoldenFlowConfig
        from goldenflow.engine.transformer import TransformEngine

        # Read all columns as strings to handle mixed-type columns (e.g., age with "forty")
        df = pl.read_csv(csv_path, infer_schema_length=0)
        engine = TransformEngine(config=GoldenFlowConfig())
        # Pass file_path so profiler bridge can use GoldenCheck when available
        result = engine.transform_df(df, source=str(csv_path))
        return result.df
