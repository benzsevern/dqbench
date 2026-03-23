# DQBench Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, tool-agnostic data quality benchmark package with three difficulty tiers, ground truth, scoring, and a CLI.

**Architecture:** Deterministic dataset generators produce CSVs + ground truth JSON per tier. A simple adapter interface lets any tool plug in. A scorer computes recall/precision/F1/FPR per tier and a composite DQBench Score. CLI orchestrates everything with Rich output.

**Tech Stack:** Python 3.11+, Polars, Typer, Rich, Pydantic 2

**Spec:** `docs/superpowers/specs/2026-03-23-dqbench-design.md`

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`, `dqbench/__init__.py`, `tests/__init__.py`, `.gitignore`, `README.md`, `LICENSE`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dqbench"
version = "1.0.0"
description = "The standard benchmark for data quality and validation tools"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{ name = "Ben Severn", email = "benzsevern@gmail.com" }]
keywords = ["data-quality", "benchmark", "data-validation", "testing", "data-engineering"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Information Analysis",
]
dependencies = [
    "polars>=1.0",
    "typer>=0.12",
    "rich>=13.0",
    "pydantic>=2.0",
]

[project.urls]
Homepage = "https://github.com/benzsevern/dqbench"
Repository = "https://github.com/benzsevern/dqbench"

[project.scripts]
dqbench = "dqbench.cli:app"

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]
goldencheck = ["goldencheck>=0.2.0"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package files**

`dqbench/__init__.py`:
```python
"""DQBench — the standard benchmark for data quality tools."""
__version__ = "1.0.0"
```

`tests/__init__.py`: empty.

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
.env
.ruff_cache/
.testing/
```

`LICENSE`: MIT, 2026, Ben Severn.

- [ ] **Step 3: Create placeholder README**

```markdown
# DQBench

The standard benchmark for data quality and validation tools.

## Install

```bash
pip install dqbench
```

## Quick Start

```bash
dqbench run goldencheck
```
```

- [ ] **Step 4: Install and verify**

```bash
cd D:/show_case/dqbench && pip install -e ".[dev]"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: initial project scaffold"
```

---

## Task 2: Data Models

**Files:**
- Create: `dqbench/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests**

```python
from dqbench.models import DQBenchFinding, TierResult, Scorecard

def test_finding_creation():
    f = DQBenchFinding(column="email", severity="error", check="format", message="bad")
    assert f.column == "email"
    assert f.confidence == 1.0

def test_tier_result():
    r = TierResult(tier=1, recall=0.9, precision=0.8, f1=0.847, false_positive_rate=0.1,
                   time_seconds=0.5, memory_mb=10, findings_count=50)
    assert r.f1 == 0.847

def test_scorecard():
    t1 = TierResult(tier=1, recall=1.0, precision=0.85, f1=0.92, false_positive_rate=0.0,
                    time_seconds=0.1, memory_mb=8, findings_count=45)
    t2 = TierResult(tier=2, recall=0.88, precision=0.72, f1=0.793, false_positive_rate=0.18,
                    time_seconds=1.4, memory_mb=45, findings_count=120)
    t3 = TierResult(tier=3, recall=0.65, precision=0.60, f1=0.624, false_positive_rate=0.22,
                    time_seconds=3.1, memory_mb=90, findings_count=200)
    sc = Scorecard(tool_name="Test", tool_version="1.0", tiers=[t1, t2, t3])
    # 0.92*0.2 + 0.793*0.4 + 0.624*0.4 = 0.184 + 0.3172 + 0.2496 = 0.7508
    assert abs(sc.dqbench_score - 75.08) < 0.1
```

- [ ] **Step 2: Implement models.py**

```python
"""DQBench data models."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DQBenchFinding:
    column: str
    severity: str
    check: str
    message: str
    confidence: float = 1.0

@dataclass
class TierResult:
    tier: int
    recall: float
    precision: float
    f1: float
    false_positive_rate: float
    time_seconds: float
    memory_mb: float
    findings_count: int

@dataclass
class Scorecard:
    tool_name: str
    tool_version: str
    tiers: list[TierResult]
    llm_cost: float | None = None
    confidence_calibration: float | None = None

    @property
    def dqbench_score(self) -> float:
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        total = 0.0
        for t in self.tiers:
            total += t.f1 * weights.get(t.tier, 0) * 100
        return round(total, 2)
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest -v && git add dqbench/models.py tests/test_models.py && git commit -m "feat: add data models"
```

---

## Task 3: Adapter Interface

**Files:**
- Create: `dqbench/adapters/__init__.py`
- Create: `dqbench/adapters/base.py`
- Create: `tests/test_adapters.py`

- [ ] **Step 1: Implement base adapter**

```python
"""Base adapter interface for validation tools."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from dqbench.models import DQBenchFinding

class DQBenchAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: ...
```

- [ ] **Step 2: Write test with a mock adapter**

```python
from pathlib import Path
from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding

class MockAdapter(DQBenchAdapter):
    @property
    def name(self): return "MockTool"
    @property
    def version(self): return "1.0"
    def validate(self, csv_path):
        return [DQBenchFinding(column="test", severity="error", check="test", message="test")]

def test_mock_adapter():
    adapter = MockAdapter()
    assert adapter.name == "MockTool"
    findings = adapter.validate(Path("fake.csv"))
    assert len(findings) == 1
```

- [ ] **Step 3: Commit**

```bash
git add dqbench/adapters/ tests/test_adapters.py && git commit -m "feat: add adapter interface"
```

---

## Task 4: Ground Truth + Scorer

**Files:**
- Create: `dqbench/ground_truth.py`
- Create: `dqbench/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Implement ground_truth.py**

```python
"""Load and query ground truth manifests."""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel

class PlantedColumn(BaseModel):
    issues: list[str]
    planted_count: int
    description: str
    affected_rows: list[int] = []

class GroundTruth(BaseModel):
    tier: int
    version: str
    rows: int
    columns: int
    planted_columns: dict[str, PlantedColumn]
    clean_columns: list[str]
    total_planted_issues: int

def load_ground_truth(path: Path) -> GroundTruth:
    with open(path) as f:
        return GroundTruth(**json.load(f))
```

- [ ] **Step 2: Implement scorer.py**

```python
"""Compute benchmark scores from findings vs ground truth."""
from __future__ import annotations
from dqbench.models import DQBenchFinding, TierResult
from dqbench.ground_truth import GroundTruth

def score_tier(
    findings: list[DQBenchFinding],
    ground_truth: GroundTruth,
    tier: int,
    time_seconds: float,
    memory_mb: float,
) -> TierResult:
    planted_cols = set(ground_truth.planted_columns.keys())
    clean_cols = set(ground_truth.clean_columns)

    # Detection: any finding on a planted column (any severity)
    detected_cols = set()
    for f in findings:
        for part in f.column.split(","):
            col = part.strip()
            if col in planted_cols:
                detected_cols.add(col)

    # False positives: WARNING/ERROR on clean columns
    fp_cols = set()
    for f in findings:
        if f.severity in ("error", "warning"):
            for part in f.column.split(","):
                col = part.strip()
                if col in clean_cols:
                    fp_cols.add(col)

    tp = len(detected_cols)
    fn = len(planted_cols) - tp
    fp = len(fp_cols)

    recall = tp / len(planted_cols) if planted_cols else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = len(fp_cols) / len(clean_cols) if clean_cols else 0

    return TierResult(
        tier=tier, recall=round(recall, 4), precision=round(precision, 4),
        f1=round(f1, 4), false_positive_rate=round(fpr, 4),
        time_seconds=round(time_seconds, 3), memory_mb=round(memory_mb, 1),
        findings_count=len(findings),
    )
```

- [ ] **Step 3: Write comprehensive scorer tests**

```python
from dqbench.scorer import score_tier
from dqbench.ground_truth import GroundTruth, PlantedColumn
from dqbench.models import DQBenchFinding

def _make_gt(planted: dict[str, list[str]], clean: list[str], tier: int = 1) -> GroundTruth:
    return GroundTruth(
        tier=tier, version="1.0", rows=100, columns=len(planted) + len(clean),
        planted_columns={k: PlantedColumn(issues=v, planted_count=1, description="test") for k, v in planted.items()},
        clean_columns=clean, total_planted_issues=sum(len(v) for v in planted.values()),
    )

def test_perfect_score():
    gt = _make_gt({"email": ["format"], "age": ["range"]}, ["notes"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="format", message="bad"),
        DQBenchFinding(column="age", severity="warning", check="range", message="outlier"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.false_positive_rate == 0.0

def test_false_positive():
    gt = _make_gt({"email": ["format"]}, ["notes", "tags"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="x", message="x"),
        DQBenchFinding(column="notes", severity="warning", check="x", message="x"),  # FP
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0
    assert result.precision == 0.5  # 1 TP, 1 FP
    assert result.false_positive_rate == 0.5  # 1 of 2 clean cols flagged

def test_info_on_clean_not_fp():
    gt = _make_gt({"email": ["format"]}, ["notes"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="x", message="x"),
        DQBenchFinding(column="notes", severity="info", check="x", message="x"),  # NOT a FP
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.false_positive_rate == 0.0

def test_missed_column():
    gt = _make_gt({"email": ["format"], "age": ["range"]}, [])
    findings = [DQBenchFinding(column="email", severity="error", check="x", message="x")]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 0.5

def test_comma_joined_column():
    gt = _make_gt({"start": ["temporal"], "end": ["temporal"]}, [])
    findings = [DQBenchFinding(column="end,start", severity="warning", check="temporal", message="x")]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0  # both columns detected via comma split
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest -v && git add dqbench/ground_truth.py dqbench/scorer.py tests/test_scorer.py && git commit -m "feat: add ground truth loader and scorer"
```

---

## Task 5: Tier 1 Generator

**Files:**
- Create: `dqbench/generator/__init__.py`
- Create: `dqbench/generator/utils.py`
- Create: `dqbench/generator/tier1.py`
- Create: `tests/test_generator_tier1.py`

- [ ] **Step 1: Implement utils.py** — shared fake data pools (names, domains, cities, states, zips). All using stdlib `random` only.

- [ ] **Step 2: Implement tier1.py** — generates 5K rows, 20 columns (15 with issues, 5 clean) per spec. Returns `(pl.DataFrame, GroundTruth)`. Uses `random.seed(42)`.

Follow the spec's Tier 1 table exactly:
- customer_id: 15 duplicates
- first_name: 8 nulls
- last_name: 5 numeric values
- email: 25 non-email values
- phone: 3 format variants
- age: 20 string values + 8 outliers
- income: 10 extreme outliers
- status: 15 misspelled variants
- signup_date: 12 wrong format
- last_login: 18 temporal violations
- country: 10 invalid codes
- zip_code: mixed formats
- shipping_address/city/zip: 50 correlated nulls
- 5 clean columns: order_count, account_type, last_updated, notes, referral_source

- [ ] **Step 3: Write test**

```python
def test_tier1_generation():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    assert len(df) == 5000
    assert len(df.columns) == 20
    assert gt.tier == 1
    assert len(gt.planted_columns) == 15
    assert len(gt.clean_columns) == 5
    assert gt.total_planted_issues > 0

def test_tier1_deterministic():
    from dqbench.generator.tier1 import generate_tier1
    df1, gt1 = generate_tier1()
    df2, gt2 = generate_tier1()
    assert df1.frame_equal(df2)
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest -v && git add dqbench/generator/ tests/test_generator_tier1.py && git commit -m "feat: add Tier 1 dataset generator"
```

---

## Task 6: Tier 2 Generator

**Files:**
- Create: `dqbench/generator/tier2.py`
- Create: `tests/test_generator_tier2.py`

- [ ] **Step 1: Implement tier2.py** — 50K rows, 30 columns (15 with issues, 15 clean). Follow spec table. Subtle issues: near-threshold outliers, gradual drift, low-frequency errors, wrong-context formats, type ambiguity.

- [ ] **Step 2: Write test**

```python
def test_tier2_generation():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert len(df) == 50000
    assert len(df.columns) == 30
    assert gt.tier == 2
    assert len(gt.planted_columns) == 15
    assert len(gt.clean_columns) == 15

def test_tier2_deterministic():
    from dqbench.generator.tier2 import generate_tier2
    df1, _ = generate_tier2()
    df2, _ = generate_tier2()
    assert df1.frame_equal(df2)
```

- [ ] **Step 3: Commit**

```bash
pytest -v && git add dqbench/generator/tier2.py tests/test_generator_tier2.py && git commit -m "feat: add Tier 2 dataset generator"
```

---

## Task 7: Tier 3 Generator

**Files:**
- Create: `dqbench/generator/tier3.py`
- Create: `tests/test_generator_tier3.py`

- [ ] **Step 1: Implement tier3.py** — 100K rows, 50 columns (25 with issues, 25 clean). Healthcare domain. Adversarial: Luhn check digit failures, cross-column logic, encoding traps, statistical traps, temporal logic.

- [ ] **Step 2: Write test**

```python
def test_tier3_generation():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    assert len(df) == 100000
    assert len(df.columns) == 50
    assert gt.tier == 3
    assert len(gt.planted_columns) == 25
    assert len(gt.clean_columns) == 25

def test_tier3_deterministic():
    from dqbench.generator.tier3 import generate_tier3
    df1, _ = generate_tier3()
    df2, _ = generate_tier3()
    assert df1.frame_equal(df2)
```

- [ ] **Step 3: Commit**

```bash
pytest -v && git add dqbench/generator/tier3.py tests/test_generator_tier3.py && git commit -m "feat: add Tier 3 dataset generator"
```

---

## Task 8: Runner + Report

**Files:**
- Create: `dqbench/runner.py`
- Create: `dqbench/report.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Implement runner.py**

```python
"""Orchestrate adapter against all tiers."""
from __future__ import annotations
import time
import tracemalloc
import json
from pathlib import Path
import polars as pl
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
            json.dump(gt.model_dump() if hasattr(gt, 'model_dump') else gt.__dict__, f, indent=2)

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

        result = score_tier(findings, gt, tier=tier_num,
                          time_seconds=elapsed, memory_mb=peak / (1024*1024))
        results.append(result)

    return Scorecard(tool_name=adapter.name, tool_version=adapter.version, tiers=results)
```

- [ ] **Step 2: Implement report.py** — Rich console scorecard matching the spec's example output format. Also JSON output mode.

- [ ] **Step 3: Write tests**

```python
from dqbench.runner import run_benchmark
from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding
from pathlib import Path

class NullAdapter(DQBenchAdapter):
    @property
    def name(self): return "NullTool"
    @property
    def version(self): return "0.0"
    def validate(self, csv_path): return []

def test_run_benchmark_returns_scorecard():
    scorecard = run_benchmark(NullAdapter(), tiers=[1])
    assert scorecard.tool_name == "NullTool"
    assert len(scorecard.tiers) == 1
    assert scorecard.tiers[0].recall == 0.0  # null adapter finds nothing
```

- [ ] **Step 4: Commit**

```bash
pytest -v && git add dqbench/runner.py dqbench/report.py tests/test_runner.py && git commit -m "feat: add benchmark runner and Rich scorecard report"
```

---

## Task 9: GoldenCheck Adapter

**Files:**
- Create: `dqbench/adapters/goldencheck.py`

- [ ] **Step 1: Implement adapter**

```python
"""GoldenCheck adapter for DQBench."""
from __future__ import annotations
from pathlib import Path
from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding

class GoldenCheckAdapter(DQBenchAdapter):
    @property
    def name(self) -> str:
        return "GoldenCheck"

    @property
    def version(self) -> str:
        try:
            from goldencheck import __version__
            return __version__
        except ImportError:
            return "not installed"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        from goldencheck.engine.scanner import scan_file
        findings, _ = scan_file(csv_path)
        return [
            DQBenchFinding(
                column=f.column,
                severity=f.severity.name.lower(),
                check=f.check,
                message=f.message,
                confidence=getattr(f, 'confidence', 1.0),
            )
            for f in findings
        ]
```

- [ ] **Step 2: Commit**

```bash
git add dqbench/adapters/goldencheck.py && git commit -m "feat: add GoldenCheck adapter"
```

---

## Task 10: CLI

**Files:**
- Create: `dqbench/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Implement CLI**

```python
"""DQBench CLI."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional
import typer
from dqbench import __version__

app = typer.Typer(name="dqbench", help="The standard benchmark for data quality tools.")

@app.command()
def run(
    adapter_name: str = typer.Argument(..., help="Adapter name (e.g., 'goldencheck') or --adapter path"),
    tier: Optional[int] = typer.Option(None, "--tier", "-t", help="Run specific tier only (1, 2, or 3)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    adapter_path: Optional[Path] = typer.Option(None, "--adapter", help="Path to custom adapter file"),
) -> None:
    """Run benchmark against a validation tool."""
    from dqbench.runner import run_benchmark

    adapter = _load_adapter(adapter_name, adapter_path)
    tiers = [tier] if tier else None
    scorecard = run_benchmark(adapter, tiers=tiers)

    if json_output:
        from dqbench.report import report_json
        report_json(scorecard, sys.stdout)
    else:
        from dqbench.report import report_rich
        report_rich(scorecard)

@app.command()
def generate(
    force: bool = typer.Option(False, "--force", help="Regenerate even if cached"),
) -> None:
    """Generate benchmark datasets."""
    from dqbench.runner import CACHE_DIR, ensure_datasets
    if force:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
    ensure_datasets()
    typer.echo(f"Datasets generated at {CACHE_DIR}")

@app.command()
def results() -> None:
    """Show results from last run."""
    typer.echo("No cached results. Run 'dqbench run <adapter>' first.")

def _load_adapter(name: str, path: Path | None = None):
    if path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_adapter", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and hasattr(obj, 'validate') and obj.__name__ != 'DQBenchAdapter':
                return obj()
        raise typer.Exit("No DQBenchAdapter subclass found in adapter file.")

    builtin = {"goldencheck": "dqbench.adapters.goldencheck:GoldenCheckAdapter"}
    if name in builtin:
        module_path, class_name = builtin[name].split(":")
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)()

    raise typer.Exit(f"Unknown adapter: {name}. Use --adapter to specify a custom adapter file.")
```

- [ ] **Step 2: Write CLI tests**

```python
from typer.testing import CliRunner
from dqbench.cli import app

runner = CliRunner()

def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dqbench" in result.stdout.lower()

def test_generate():
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest -v && ruff check . && git add dqbench/cli.py tests/test_cli.py && git commit -m "feat: add CLI with run, generate, and results commands"
```

---

## Task 11: README + Docs + GitHub

**Files:**
- Modify: `README.md`
- Create: `CONTRIBUTING.md`
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write full README** with: what it is, install, quick start, adapter interface example, scoring methodology, tier descriptions, CLI reference, how to write a custom adapter

- [ ] **Step 2: Create CONTRIBUTING.md** with: how to add adapters, how to propose tier changes, dev setup

- [ ] **Step 3: Create CI workflow**

- [ ] **Step 4: Create GitHub repo and push**

```bash
gh auth switch --user benzsevern
gh repo create benzsevern/dqbench --public --source=. --description "The standard benchmark for data quality and validation tools" --push
```

- [ ] **Step 5: Set topics and create release**

```bash
gh repo edit benzsevern/dqbench --add-topic data-quality --add-topic benchmark --add-topic data-validation --add-topic python --add-topic data-engineering
gh release create v1.0.0 --title "v1.0.0 — Initial Release" --notes "Three-tier data quality benchmark with scoring framework."
```

- [ ] **Step 6: Publish to PyPI**

```bash
python -m build && source .testing/.env && python -m twine upload dist/*
```

---

## Task 12: Run GoldenCheck Against DQBench

- [ ] **Step 1: Run the full benchmark**

```bash
dqbench run goldencheck
```

- [ ] **Step 2: Capture and save results**

- [ ] **Step 3: Update GoldenCheck README** with DQBench scores

- [ ] **Step 4: Commit DQBench results to GoldenCheck repo**
