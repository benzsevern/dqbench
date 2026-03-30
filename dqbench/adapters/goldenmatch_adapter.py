"""GoldenMatch adapter for DQBench ER benchmarks."""
from __future__ import annotations
from pathlib import Path
from dqbench.adapters.base import EntityResolutionAdapter


class GoldenMatchAdapter(EntityResolutionAdapter):
    @property
    def name(self) -> str:
        return "goldenmatch"

    @property
    def version(self) -> str:
        try:
            import goldenmatch
            return goldenmatch.__version__
        except ImportError:
            return "not-installed"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        try:
            import goldenmatch
            import polars as pl
        except ImportError:
            raise RuntimeError(
                "goldenmatch is not installed. Run: pip install goldenmatch"
            )

        df = pl.read_csv(csv_path)

        # Match on all identity columns with appropriate thresholds.
        # Names get lower threshold (typos, nicknames, phonetics).
        # Email/phone get higher (structured, mostly exact or clearly different).
        # Block on last_name to keep comparison count manageable.
        # All fuzzy, no exact — exact matching on email creates
        # oversized clusters with common emails in synthetic data.
        # Higher threshold to control false positives.
        result = goldenmatch.dedupe_df(
            df,
            fuzzy={
                "first_name": 0.70,
                "last_name": 0.80,
                "email": 0.85,
                "phone": 0.75,
                "address": 0.75,
                "city": 0.90,
            },
            threshold=0.65,
        )

        # Extract pairs from clusters (the correct way).
        # Each cluster has a "members" list of __row_id__ values.
        pairs = []
        if result.clusters:
            for cluster in result.clusters.values():
                members = sorted(cluster["members"])
                if len(members) < 2:
                    continue
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        pairs.append((members[i], members[j]))

        return pairs
