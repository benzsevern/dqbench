"""GoldenMatch adapter for DQBench ER benchmarks."""
from __future__ import annotations
from pathlib import Path
from dqbench.adapters.base import EntityResolutionAdapter


class GoldenMatchAdapter(EntityResolutionAdapter):
    @property
    def name(self) -> str: return "goldenmatch"

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
            raise RuntimeError("goldenmatch is not installed. Run: pip install goldenmatch")

        df = pl.read_csv(csv_path)

        # Use explicit fuzzy config on key identity columns
        # Email as exact anchor, names + address as fuzzy
        result = goldenmatch.dedupe_df(
            df,
            fuzzy={"first_name": 0.8, "last_name": 0.8},
            exact=["email"],
        )

        # Extract duplicate pairs grouped by matchkey cluster
        pairs = []
        if result.dupes.shape[0] > 0 and "__row_id__" in result.dupes.columns:
            mk_cols = [c for c in result.dupes.columns if c.startswith("__mk_")]
            if mk_cols:
                mk_col = mk_cols[0]
                clusters = result.dupes.group_by(mk_col).agg(
                    pl.col("__row_id__").alias("row_ids")
                )
                for row in clusters.iter_rows(named=True):
                    ids = sorted(row["row_ids"])
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            pairs.append((ids[i], ids[j]))

        return pairs
