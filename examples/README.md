# DQBench Examples

## Quick Start

```bash
pip install dqbench
```

### Benchmark Your Own Tool

```bash
python examples/benchmark_your_tool.py
```

Implements all 4 adapter types with placeholder logic. Replace each adapter's method with your tool's API to get your DQBench scores.

### Run the Golden Suite

```bash
pip install goldencheck goldenflow goldenmatch
python examples/golden_suite_benchmark.py
```

Runs all 4 Golden Suite tools across all benchmark categories. ~30s without LLM, ~12min with LLM.

### Individual Benchmarks

| Script | Category | Prerequisites | Time |
|--------|----------|--------------|------|
| `run_benchmark.py` | Detect | `pip install goldencheck` | ~10s |
| `run_er_benchmark.py` | ER (baseline) | none | ~5s |
| `run_pipeline_benchmark.py` | Pipeline (baseline) | none | ~5s |
| `run_goldenmatch_benchmark.py` | ER (GoldenMatch) | `pip install goldenmatch` | ~23s |

### GitHub Actions

Run benchmarks directly from the Actions tab:

| Workflow | What it runs | Trigger |
|----------|-------------|---------|
| **Try DQBench ER** | GoldenMatch ER benchmark | Manual (workflow_dispatch) |
| **Try DQBench Pipeline** | GoldenPipe Pipeline benchmark | Manual (workflow_dispatch) |
| **Try DQBench (All)** | All 4 categories in parallel | Manual (workflow_dispatch) |

### Cost Estimates

| Benchmark | Without LLM | With LLM |
|-----------|-------------|----------|
| Detect (GoldenCheck) | Free, ~10s | N/A |
| Transform (GoldenFlow) | Free, ~5s | N/A |
| ER (GoldenMatch) | Free, ~23s | ~$0.25, ~670s |
| Pipeline (GoldenPipe) | Free, ~28s | ~$0.50, ~20min |
| **Full Suite** | **Free, ~66s** | **~$0.75, ~35min** |
