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
            from goldenmatch.core.engine import MatchEngine
        except ImportError:
            raise RuntimeError("goldenmatch is not installed. Run: pip install goldenmatch")
        engine = MatchEngine()
        result = engine.dedupe_file(csv_path)
        pairs = []
        for cluster in result.clusters:
            members = sorted(cluster.member_ids)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.append((members[i], members[j]))
        return pairs
