"""Orchestrate adapter against all tiers."""
from __future__ import annotations
import time
import tracemalloc
import json
from pathlib import Path

from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import Scorecard
from dqbench.ground_truth import load_ground_truth
from dqbench.scorer import score_tier

CACHE_DIR = Path.home() / ".dqbench" / "datasets"


def ensure_datasets() -> None:
    """Generate datasets if not cached."""
    if (CACHE_DIR / "tier1" / "data.csv").exists():
        return
    from dqbench.generator.tier1 import generate_tier1
    from dqbench.generator.tier2 import generate_tier2
    from dqbench.generator.tier3 import generate_tier3

    for tier_num, gen_fn in [(1, generate_tier1), (2, generate_tier2), (3, generate_tier3)]:
        tier_dir = CACHE_DIR / f"tier{tier_num}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        df, gt = gen_fn()
        df.write_csv(tier_dir / "data.csv")
        with open(tier_dir / "ground_truth.json", "w") as f:
            json.dump(gt.model_dump() if hasattr(gt, "model_dump") else gt.__dict__, f, indent=2)


def run_benchmark(
    adapter: DQBenchAdapter,
    tiers: list[int] | None = None,
) -> Scorecard:
    """Run the benchmark and return a scorecard."""
    ensure_datasets()
    tier_nums = tiers or [1, 2, 3]
    results = []

    for tier_num in tier_nums:
        tier_dir = CACHE_DIR / f"tier{tier_num}"
        csv_path = tier_dir / "data.csv"
        gt = load_ground_truth(tier_dir / "ground_truth.json")

        tracemalloc.start()
        t0 = time.perf_counter()
        findings = adapter.validate(csv_path)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = score_tier(
            findings,
            gt,
            tier=tier_num,
            time_seconds=elapsed,
            memory_mb=peak / (1024 * 1024),
        )
        results.append(result)

    return Scorecard(tool_name=adapter.name, tool_version=adapter.version, tiers=results)
