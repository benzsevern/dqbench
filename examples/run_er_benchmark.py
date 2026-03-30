"""Example: Run the DQBench ER benchmark with a custom adapter.

Usage:
    pip install dqbench
    python examples/run_er_benchmark.py

This example implements a simple baseline ER adapter using basic
string matching. Replace the deduplicate() method with your own
tool to benchmark it.
"""
from __future__ import annotations
from pathlib import Path
from dqbench.adapters.base import EntityResolutionAdapter
from dqbench.runner import run_er_benchmark
from dqbench.report import report_er_rich


class SimpleERAdapter(EntityResolutionAdapter):
    """Baseline ER adapter using exact email matching.

    Replace this with your own entity resolution logic to benchmark it.
    """
    @property
    def name(self) -> str:
        return "simple-baseline"

    @property
    def version(self) -> str:
        return "1.0.0"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        import polars as pl

        df = pl.read_csv(csv_path)

        # Simple approach: records with identical emails are duplicates
        pairs = []
        email_groups: dict[str, list[int]] = {}
        for i, email in enumerate(df["email"].to_list()):
            key = email.strip().lower() if email else ""
            if key:
                email_groups.setdefault(key, []).append(i)

        for ids in email_groups.values():
            if len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.append((ids[i], ids[j]))

        return pairs


if __name__ == "__main__":
    adapter = SimpleERAdapter()
    scorecard = run_er_benchmark(adapter)
    report_er_rich(scorecard)
    print(f"\nDQBench ER Score: {scorecard.dqbench_er_score:.2f} / 100")
