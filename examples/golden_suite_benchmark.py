"""Run DQBench across the entire Golden Suite.

Usage:
    pip install dqbench goldencheck goldenflow goldenmatch
    python examples/golden_suite_benchmark.py

For best ER results, set OPENAI_API_KEY for LLM scoring.
Estimated cost with LLM: ~$0.25. Without LLM: free.
"""
from __future__ import annotations
import time

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


def main():
    results = {}
    total_start = time.perf_counter()

    # -- GoldenCheck (Detect) --
    print("\n" + "=" * 60)
    print("GoldenCheck — Detect Benchmark")
    print("=" * 60)
    from dqbench.adapters.goldencheck import GoldenCheckAdapter
    sc = run_benchmark(GoldenCheckAdapter())
    report_rich(sc)
    results["Detect"] = sc.dqbench_score

    # -- GoldenFlow (Transform) --
    print("\n" + "=" * 60)
    print("GoldenFlow — Transform Benchmark")
    print("=" * 60)
    from dqbench.adapters.goldenflow import GoldenFlowAdapter
    sc = run_transform_benchmark(GoldenFlowAdapter())
    report_transform_rich(sc)
    results["Transform"] = sc.composite_score

    # -- GoldenMatch (ER) --
    print("\n" + "=" * 60)
    print("GoldenMatch — ER Benchmark")
    print("=" * 60)
    from dqbench.adapters.goldenmatch_adapter import GoldenMatchAdapter
    sc = run_er_benchmark(GoldenMatchAdapter())
    report_er_rich(sc)
    results["ER"] = sc.dqbench_er_score

    # -- GoldenPipe (Pipeline) --
    print("\n" + "=" * 60)
    print("GoldenPipe — Pipeline Benchmark")
    print("=" * 60)
    from dqbench.adapters.goldenpipe_adapter import GoldenPipeAdapter
    sc = run_pipeline_benchmark(GoldenPipeAdapter())
    report_pipeline_rich(sc)
    results["Pipeline"] = sc.dqbench_pipeline_score

    total_time = time.perf_counter() - total_start

    # -- Summary --
    print("\n" + "=" * 60)
    print("Golden Suite — DQBench Summary")
    print("=" * 60)
    for category, score in results.items():
        print(f"  {category:12s}: {score:6.2f} / 100")
    print(f"\n  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
