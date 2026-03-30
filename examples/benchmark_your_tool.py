"""Benchmark your own data quality tool against DQBench.

This script shows how to implement all 4 adapter types and run each benchmark.
Replace the placeholder logic with your tool's actual API calls.

Usage:
    pip install dqbench
    python examples/benchmark_your_tool.py
"""
from __future__ import annotations
from pathlib import Path
import polars as pl

from dqbench.adapters.base import (
    DQBenchAdapter,
    TransformAdapter,
    EntityResolutionAdapter,
    PipelineAdapter,
)
from dqbench.models import DQBenchFinding


# -- 1. Detection Adapter ------------------------------------------------
class MyDetectAdapter(DQBenchAdapter):
    """Replace with your validation tool."""

    @property
    def name(self) -> str:
        return "my-detect-tool"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        df = pl.read_csv(csv_path)
        findings = []
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                findings.append(DQBenchFinding(
                    column=col,
                    severity="warning",
                    check="nullability",
                    message=f"{null_count} null values found",
                ))
        return findings


# -- 2. Transform Adapter ------------------------------------------------
class MyTransformAdapter(TransformAdapter):
    """Replace with your data cleaning tool."""

    @property
    def name(self) -> str:
        return "my-transform-tool"

    @property
    def version(self) -> str:
        return "1.0.0"

    def transform(self, csv_path: Path) -> pl.DataFrame:
        df = pl.read_csv(csv_path)
        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
        return df


# -- 3. Entity Resolution Adapter ----------------------------------------
class MyERAdapter(EntityResolutionAdapter):
    """Replace with your deduplication tool."""

    @property
    def name(self) -> str:
        return "my-er-tool"

    @property
    def version(self) -> str:
        return "1.0.0"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        df = pl.read_csv(csv_path)
        pairs = []
        # Simple: match records with identical lowercased emails
        if "email" in df.columns:
            groups: dict[str, list[int]] = {}
            for i, email in enumerate(df["email"].to_list()):
                key = str(email).strip().lower()
                if key and key != "none":
                    groups.setdefault(key, []).append(i)
            for ids in groups.values():
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.append((ids[i], ids[j]))
        return pairs


# -- 4. Pipeline Adapter --------------------------------------------------
class MyPipelineAdapter(PipelineAdapter):
    """Replace with your end-to-end pipeline."""

    @property
    def name(self) -> str:
        return "my-pipeline-tool"

    @property
    def version(self) -> str:
        return "1.0.0"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        df = pl.read_csv(csv_path)
        # Step 1: Clean -- strip whitespace
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
        # Step 2: Dedupe -- remove exact duplicates by email
        if "email" in df.columns:
            df = df.with_columns(
                pl.col("email").str.to_lowercase().alias("_dedup_key")
            )
            df = df.unique(subset=["_dedup_key"], keep="first").drop("_dedup_key")
        return df


# -- Run All Benchmarks ---------------------------------------------------
if __name__ == "__main__":
    from dqbench.runner import (
        run_benchmark,
        run_transform_benchmark,
        run_er_benchmark,
        run_pipeline_benchmark,
    )
    from dqbench.report import (
        report_rich,
        report_transform_rich,
        report_er_rich,
        report_pipeline_rich,
    )

    print("=" * 60)
    print("DQBench — Benchmark Your Tool")
    print("=" * 60)

    print("\n-- Detect --")
    sc = run_benchmark(MyDetectAdapter())
    report_rich(sc)
    print(f"DQBench Detect Score: {sc.dqbench_score:.2f}")

    print("\n-- Transform --")
    sc = run_transform_benchmark(MyTransformAdapter())
    report_transform_rich(sc)
    print(f"DQBench Transform Score: {sc.composite_score:.2f}")

    print("\n-- Entity Resolution --")
    sc = run_er_benchmark(MyERAdapter())
    report_er_rich(sc)
    print(f"DQBench ER Score: {sc.dqbench_er_score:.2f}")

    print("\n-- Pipeline --")
    sc = run_pipeline_benchmark(MyPipelineAdapter())
    report_pipeline_rich(sc)
    print(f"DQBench Pipeline Score: {sc.dqbench_pipeline_score:.2f}")
