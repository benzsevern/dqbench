"""Tests for adapter base classes and inheritance."""
from __future__ import annotations
from pathlib import Path
import polars as pl

from dqbench.adapters.base import (
    BenchmarkAdapter,
    DQBenchAdapter,
    TransformAdapter,
    EntityResolutionAdapter,
    PipelineAdapter,
)
from dqbench.models import DQBenchFinding


class FakeDetectAdapter(DQBenchAdapter):
    @property
    def name(self) -> str: return "fake-detect"
    @property
    def version(self) -> str: return "0.1.0"
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: return []


class FakeTransformAdapter(TransformAdapter):
    @property
    def name(self) -> str: return "fake-transform"
    @property
    def version(self) -> str: return "0.1.0"
    def transform(self, csv_path: Path) -> pl.DataFrame: return pl.DataFrame()


class FakeERAdapter(EntityResolutionAdapter):
    @property
    def name(self) -> str: return "fake-er"
    @property
    def version(self) -> str: return "0.1.0"
    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]: return []


class FakePipelineAdapter(PipelineAdapter):
    @property
    def name(self) -> str: return "fake-pipeline"
    @property
    def version(self) -> str: return "0.1.0"
    def run_pipeline(self, csv_path: Path) -> pl.DataFrame: return pl.DataFrame()


def test_all_adapters_inherit_benchmark_adapter():
    for adapter in [FakeDetectAdapter(), FakeTransformAdapter(), FakeERAdapter(), FakePipelineAdapter()]:
        assert isinstance(adapter, BenchmarkAdapter)


def test_adapter_names():
    assert FakeDetectAdapter().name == "fake-detect"
    assert FakeERAdapter().name == "fake-er"
    assert FakePipelineAdapter().name == "fake-pipeline"


def test_existing_adapters_still_work():
    d = FakeDetectAdapter()
    assert d.validate(Path("x.csv")) == []
    t = FakeTransformAdapter()
    assert t.transform(Path("x.csv")).shape == (0, 0)
