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
        from goldenflow import transform_file
        result = transform_file(csv_path)
        return result.df
