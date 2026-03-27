"""Run the DQBench benchmark using the built-in GoldenCheck adapter.

Generates tier 1/2/3 datasets (cached in ~/.dqbench/datasets/), runs
GoldenCheck against each tier, and prints the scorecard.

Usage:
    python run_benchmark.py
"""
from dqbench.runner import run_benchmark
from dqbench.report import print_scorecard
from dqbench.adapters.goldencheck import GoldenCheckAdapter


def main():
    adapter = GoldenCheckAdapter()
    print(f"Running DQBench with {adapter.name} v{adapter.version}...\n")

    scorecard = run_benchmark(adapter)

    print_scorecard(scorecard)
    print(f"\nDQBench Score: {scorecard.dqbench_score:.2f} / 100")


if __name__ == "__main__":
    main()
