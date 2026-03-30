"""Run the DQBench ER benchmark using GoldenMatch.

Usage:
    pip install dqbench goldenmatch
    python examples/run_goldenmatch_benchmark.py

For best results, set OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM scoring.
Without LLM: DQBench ER ~77. With LLM: DQBench ER ~95.

Estimated cost with LLM scoring: ~$0.15-0.30 per full run (3 tiers).
"""
from dqbench.adapters.goldenmatch_adapter import GoldenMatchAdapter
from dqbench.runner import run_er_benchmark
from dqbench.report import report_er_rich

if __name__ == "__main__":
    adapter = GoldenMatchAdapter()
    scorecard = run_er_benchmark(adapter)
    report_er_rich(scorecard)
    print(f"\nDQBench ER Score: {scorecard.dqbench_er_score:.2f} / 100")
    if scorecard.tiers:
        total_time = sum(t.time_seconds for t in scorecard.tiers)
        print(f"Total time: {total_time:.1f}s")
