"""Base adapter interface for validation tools."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl
from dqbench.models import DQBenchFinding


class DQBenchAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: ...


class TransformAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def transform(self, csv_path: Path) -> pl.DataFrame:
        """Transform the messy CSV and return the cleaned DataFrame."""
        ...
