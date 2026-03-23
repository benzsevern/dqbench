# DQBench

The standard benchmark for data quality and validation tools.

[![PyPI](https://img.shields.io/pypi/v/dqbench?color=d4a017)](https://pypi.org/project/dqbench/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> The ImageNet of data quality — standardized benchmarks for validation tools.

## Why DQBench?

Every data validation tool claims to be the best. But there's no standard way to compare them. DQBench fixes that with:

- **Three difficulty tiers** — basics, realistic, and adversarial
- **Ground truth** — every planted issue is documented with affected rows
- **Fair scoring** — recall AND precision matter (no gaming by flagging everything)
- **One number** — DQBench Score (0-100) for easy comparison
- **20-line integration** — implement one method to benchmark any tool

## Install

```bash
pip install dqbench
```

## Quick Start

```bash
# Run with built-in GoldenCheck adapter
pip install goldencheck
dqbench run goldencheck

# Run with a custom adapter
dqbench run --adapter my_adapter.py
```

## Head-to-Head Results (DQBench v1.0)

| Tool | Mode | T1 F1 | T2 F1 | T3 F1 | Score |
|------|------|-------|-------|-------|-------|
| **GoldenCheck** | **zero-config** | **84.9%** | **80.0%** | **57.6%** | **72.00** |
| Pandera | best-effort | 36.4% | 38.1% | 25.0% | 32.51 |
| Soda Core | best-effort | 38.1% | 23.5% | 13.3% | 22.36 |
| Great Expectations | best-effort | 36.4% | 23.5% | 12.5% | 21.68 |
| Great Expectations | auto-profiled | 22.2% | 42.1% | 0.0% | 21.29 |
| Soda Core | auto-profiled | 0.0% | 11.1% | 6.2% | 6.94 |
| All tools | zero-config | 0.0% | 0.0% | 0.0% | 0.00 |

> GoldenCheck's zero-config discovery outperforms every competitor's hand-written rules.

Run the comparison yourself:
```bash
pip install dqbench goldencheck great_expectations pandera soda-core
dqbench run all
```

## Tiers

| Tier | Rows | Columns | Domain | Difficulty |
|------|------|---------|--------|------------|
| **1 — Basics** | 5,000 | 20 | Customer DB | Obvious errors, baseline |
| **2 — Realistic** | 50,000 | 30 | E-commerce | Subtle issues + false positive traps |
| **3 — Adversarial** | 100,000 | 50 | Healthcare | Encoding traps, semantic errors, cross-column logic |

Each tier has columns WITH planted issues and columns WITHOUT (false positive traps). Tools that flag clean columns lose precision points.

## Scoring

| Metric | Description |
|--------|-------------|
| **Recall** | % of planted-issue columns detected |
| **Precision** | % of flagged columns that actually have issues |
| **F1** | Harmonic mean of recall and precision |
| **FPR** | Clean columns incorrectly flagged (WARNING/ERROR only) |
| **DQBench Score** | Tier1_F1 × 20% + Tier2_F1 × 40% + Tier3_F1 × 40% |

## Write Your Own Adapter

Implement one class to benchmark any tool:

```python
from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding
from pathlib import Path

class MyToolAdapter(DQBenchAdapter):
    @property
    def name(self) -> str:
        return "MyTool"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        # Run your tool on the CSV
        # Return a list of DQBenchFinding objects
        return [
            DQBenchFinding(
                column="email",
                severity="error",      # "error", "warning", or "info"
                check="format",         # what kind of issue
                message="Invalid email format",
                confidence=0.9,         # optional, 0.0-1.0
            )
        ]
```

Then run:
```bash
dqbench run --adapter my_adapter.py
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `dqbench run <adapter>` | Run benchmark |
| `dqbench run --adapter <path>` | Run with custom adapter file |
| `dqbench run <adapter> --tier 2` | Run specific tier only |
| `dqbench run <adapter> --json` | JSON output |
| `dqbench generate` | Generate/cache datasets |
| `dqbench generate --force` | Regenerate datasets |

## Built-in Adapters

| Adapter | Tool | Modes | Install |
|---------|------|-------|---------|
| `goldencheck` | GoldenCheck | zero-config | `pip install goldencheck` |
| `gx-zero`, `gx-auto`, `gx-best` | Great Expectations | zero / auto / best-effort | `pip install great_expectations` |
| `pandera-zero`, `pandera-auto`, `pandera-best` | Pandera | zero / auto / best-effort | `pip install pandera` |
| `soda-zero`, `soda-auto`, `soda-best` | Soda Core | zero / auto / best-effort | `pip install soda-core` |

Want to add your tool? See [CONTRIBUTING.md](CONTRIBUTING.md).

## Reproducibility

- Datasets are generated deterministically (`random.seed(42)`, stdlib only)
- Canonical datasets committed as release artifacts
- Version-locked: published benchmark versions are immutable

## License

MIT

---

**From the maker of [GoldenCheck](https://github.com/benzsevern/goldencheck) and [GoldenMatch](https://github.com/benzsevern/goldenmatch).**
