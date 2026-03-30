# DQBench Examples

## Validation Benchmarks

| Script | Description |
|--------|-------------|
| `run_benchmark.py` | Run the DQBench validation benchmark with the GoldenCheck adapter |
| `custom_adapter.py` | Implement and run a custom validation adapter |

## ER (Entity Resolution) Benchmarks

| Script | Description |
|--------|-------------|
| `run_er_benchmark.py` | Benchmark a custom ER adapter (simple email-matching baseline) |
| `run_goldenmatch_benchmark.py` | Benchmark GoldenMatch against all 3 ER tiers |

## Pipeline Benchmarks

| Script | Description |
|--------|-------------|
| `run_pipeline_benchmark.py` | Benchmark a custom pipeline adapter (clean + deduplicate) |

## Prerequisites

```bash
# Core (required for all examples)
pip install dqbench

# Validation examples
pip install goldencheck

# GoldenMatch ER example
pip install goldenmatch
```

## Quick Start

### Benchmark your own ER tool

1. Copy `run_er_benchmark.py`
2. Replace `SimpleERAdapter.deduplicate()` with your own logic
3. Run the script -- it generates tier 1/2/3 datasets automatically and prints a scorecard

```bash
python examples/run_er_benchmark.py
```

### Benchmark your own pipeline tool

1. Copy `run_pipeline_benchmark.py`
2. Replace `SimplePipelineAdapter.run_pipeline()` with your own logic
3. Run the script

```bash
python examples/run_pipeline_benchmark.py
```
