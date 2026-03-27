"""Implement a custom DQBench adapter and run the benchmark.

Shows how to wrap any validation tool so DQBench can score it.
The adapter must implement: name, version, and validate(csv_path).

Usage:
    python custom_adapter.py
"""
from pathlib import Path

from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding
from dqbench.runner import run_benchmark
from dqbench.report import print_scorecard


class NullCheckAdapter(DQBenchAdapter):
    """Minimal adapter that only checks for null values."""

    @property
    def name(self) -> str:
        return "NullCheck"

    @property
    def version(self) -> str:
        return "0.1.0"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        import polars as pl

        df = pl.read_csv(csv_path, ignore_errors=True)
        findings = []
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                pct = null_count / len(df) * 100
                findings.append(DQBenchFinding(
                    column=col,
                    severity="WARNING",
                    check="null_values",
                    message=f"{null_count} nulls ({pct:.1f}%)",
                    confidence=min(pct / 10, 1.0),
                ))
        return findings


def main():
    adapter = NullCheckAdapter()
    scorecard = run_benchmark(adapter)
    print_scorecard(scorecard)
    print(f"\nDQBench Score: {scorecard.dqbench_score:.2f} / 100")


if __name__ == "__main__":
    main()
