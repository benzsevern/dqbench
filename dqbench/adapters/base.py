"""Base adapter interface for validation tools."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl
from dqbench.models import DQBenchFinding, OCRCompanyPrediction


class BenchmarkAdapter(ABC):
    """Shared base for all adapter types."""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...


class DQBenchAdapter(BenchmarkAdapter):
    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: ...


class TransformAdapter(BenchmarkAdapter):
    @abstractmethod
    def transform(self, csv_path: Path) -> pl.DataFrame:
        """Transform the messy CSV and return the cleaned DataFrame."""
        ...


class EntityResolutionAdapter(BenchmarkAdapter):
    @abstractmethod
    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        """Return list of (row_a, row_b) matched pairs. 0-based row indices."""
        ...


class PipelineAdapter(BenchmarkAdapter):
    @abstractmethod
    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        """Run full pipeline (validate -> transform -> deduplicate).
        Return the final cleaned, deduplicated DataFrame."""
        ...


class OCRCompanyAdapter(BenchmarkAdapter):
    @abstractmethod
    def score_companies(self, csv_path: Path) -> list[OCRCompanyPrediction]:
        """Score OCR'd company names in the benchmark CSV."""
        ...
