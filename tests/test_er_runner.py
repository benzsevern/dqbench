"""Tests for ER runner integration."""
from __future__ import annotations
from pathlib import Path


from dqbench.adapters.base import EntityResolutionAdapter
from dqbench.models import ERScorecard


class PerfectERAdapter(EntityResolutionAdapter):
    """Returns exact ground truth pairs for testing."""

    def __init__(self, pairs: list[tuple[int, int]] | None = None):
        self._pairs = pairs or []

    @property
    def name(self) -> str:
        return "perfect-er"

    @property
    def version(self) -> str:
        return "0.0.1"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        if self._pairs:
            return self._pairs
        # Load ground truth to return perfect predictions
        import json
        gt_path = csv_path.parent / "er_ground_truth.json"
        with open(gt_path) as f:
            data = json.load(f)
        return [tuple(p) for p in data["duplicate_pairs"]]


class TestERRunner:
    def test_run_er_benchmark_returns_scorecard(self):
        from dqbench.runner import run_er_benchmark, ensure_er_datasets
        ensure_er_datasets()
        adapter = PerfectERAdapter()
        scorecard = run_er_benchmark(adapter, tiers=[1])
        assert isinstance(scorecard, ERScorecard)
        assert scorecard.tool_name == "perfect-er"
        assert len(scorecard.tiers) == 1

    def test_perfect_er_scores_100(self):
        from dqbench.runner import run_er_benchmark, ensure_er_datasets
        ensure_er_datasets()
        adapter = PerfectERAdapter()
        scorecard = run_er_benchmark(adapter, tiers=[1])
        assert scorecard.tiers[0].f1 == 1.0
        assert scorecard.tiers[0].precision == 1.0
        assert scorecard.tiers[0].recall == 1.0

    def test_ensure_er_datasets_creates_files(self):
        from dqbench.runner import CACHE_DIR, ensure_er_datasets
        ensure_er_datasets()
        assert (CACHE_DIR / "er_tier1" / "data.csv").exists()
        assert (CACHE_DIR / "er_tier1" / "er_ground_truth.json").exists()
