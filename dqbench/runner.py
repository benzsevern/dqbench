"""Orchestrate adapter against all tiers."""
from __future__ import annotations
import time
import tracemalloc
import json
from pathlib import Path

import polars as pl

from dqbench.adapters.base import DQBenchAdapter, TransformAdapter, EntityResolutionAdapter, PipelineAdapter
from dqbench.models import Scorecard, TransformScorecard, ERScorecard, PipelineScorecard
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


def ensure_clean_datasets() -> None:
    """Generate clean ground truth CSVs if not cached."""
    if (CACHE_DIR / "tier1" / "data_clean.csv").exists():
        return
    from dqbench.generator.clean import generate_clean_csvs
    generate_clean_csvs(CACHE_DIR)


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


def run_transform_benchmark(
    adapter: TransformAdapter,
    tiers: list[int] | None = None,
) -> TransformScorecard:
    """Run transform benchmark and return a scorecard."""
    from dqbench.transform_scorer import score_transform_tier

    ensure_datasets()
    ensure_clean_datasets()
    tier_nums = tiers or [1, 2, 3]
    results = []

    for tier_num in tier_nums:
        tier_dir = CACHE_DIR / f"tier{tier_num}"
        csv_path = tier_dir / "data.csv"
        clean_path = tier_dir / "data_clean.csv"

        if not clean_path.exists():
            continue

        tracemalloc.start()
        t0 = time.perf_counter()
        result_df = adapter.transform(csv_path)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        clean_df = pl.read_csv(clean_path, infer_schema_length=0)
        messy_df = pl.read_csv(csv_path, infer_schema_length=0)

        tier_result = score_transform_tier(
            result_df, clean_df, messy_df, tier=tier_num,
            time_seconds=elapsed, memory_mb=peak / (1024 * 1024),
        )
        results.append(tier_result)

    return TransformScorecard(
        tool_name=adapter.name,
        tool_version=adapter.version,
        tiers=results,
    )


def ensure_er_datasets() -> None:
    """Generate ER datasets if not cached."""
    if (CACHE_DIR / "er_tier1" / "data.csv").exists():
        return
    from dqbench.generator.er_tier1 import generate_er_tier1
    from dqbench.generator.er_tier2 import generate_er_tier2
    from dqbench.generator.er_tier3 import generate_er_tier3

    generators = [
        (1, generate_er_tier1),
        (2, generate_er_tier2),
        (3, generate_er_tier3),
    ]

    for tier_num, gen_fn in generators:
        tier_dir = CACHE_DIR / f"er_tier{tier_num}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        df, gt = gen_fn()
        df.write_csv(tier_dir / "data.csv")
        with open(tier_dir / "er_ground_truth.json", "w") as f:
            json.dump(gt.model_dump(), f, indent=2)


def run_er_benchmark(
    adapter: EntityResolutionAdapter,
    tiers: list[int] | None = None,
    real: bool = False,
) -> ERScorecard:
    """Run ER benchmark and return a scorecard."""
    from dqbench.er_scorer import score_er_tier
    from dqbench.er_ground_truth import load_er_ground_truth

    ensure_er_datasets()
    tier_nums = tiers or [1, 2, 3]
    results = []

    for tier_num in tier_nums:
        tier_dir = CACHE_DIR / f"er_tier{tier_num}"
        csv_path = tier_dir / "data.csv"
        gt_path = tier_dir / "er_ground_truth.json"

        if not csv_path.exists():
            continue

        gt = load_er_ground_truth(gt_path)

        tracemalloc.start()
        t0 = time.perf_counter()
        predictions = adapter.deduplicate(csv_path)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = score_er_tier(
            predictions, gt, tier=tier_num,
            time_seconds=elapsed, memory_mb=peak / (1024 * 1024),
        )
        results.append(result)

    return ERScorecard(
        tool_name=adapter.name,
        tool_version=adapter.version,
        tiers=results,
        real_datasets=None,
    )


def ensure_pipeline_datasets() -> None:
    """Generate Pipeline datasets if not cached."""
    if (CACHE_DIR / "pipeline_tier1" / "data.csv").exists():
        return
    from dqbench.generator.pipeline_tier1 import generate_pipeline_tier1

    generators = [(1, generate_pipeline_tier1)]

    try:
        from dqbench.generator.pipeline_tier2 import generate_pipeline_tier2
        generators.append((2, generate_pipeline_tier2))
    except ImportError:
        pass
    try:
        from dqbench.generator.pipeline_tier3 import generate_pipeline_tier3
        generators.append((3, generate_pipeline_tier3))
    except ImportError:
        pass

    for tier_num, gen_fn in generators:
        tier_dir = CACHE_DIR / f"pipeline_tier{tier_num}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        messy_df, clean_df, gt = gen_fn()
        messy_df.write_csv(tier_dir / "data.csv")
        clean_df.write_csv(tier_dir / "data_clean_deduped.csv")
        with open(tier_dir / "pipeline_ground_truth.json", "w") as f:
            json.dump(gt.model_dump(), f, indent=2)


def run_pipeline_benchmark(
    adapter: PipelineAdapter,
    tiers: list[int] | None = None,
) -> PipelineScorecard:
    """Run Pipeline benchmark and return a scorecard."""
    from dqbench.pipeline_scorer import score_pipeline_tier
    from dqbench.pipeline_ground_truth import load_pipeline_ground_truth

    ensure_pipeline_datasets()
    tier_nums = tiers or [1, 2, 3]
    results = []

    for tier_num in tier_nums:
        tier_dir = CACHE_DIR / f"pipeline_tier{tier_num}"
        csv_path = tier_dir / "data.csv"
        gt_path = tier_dir / "pipeline_ground_truth.json"

        if not csv_path.exists():
            continue

        gt = load_pipeline_ground_truth(gt_path)

        tracemalloc.start()
        t0 = time.perf_counter()
        result_df = adapter.run_pipeline(csv_path)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tier_result = score_pipeline_tier(
            result_df, tier_dir, tier=tier_num,
            time_seconds=elapsed, memory_mb=peak / (1024 * 1024),
            expected_rows=gt.expected_output_rows,
        )
        results.append(tier_result)

    return PipelineScorecard(
        tool_name=adapter.name,
        tool_version=adapter.version,
        tiers=results,
    )
