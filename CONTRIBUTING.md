# Contributing to DQBench

Thank you for your interest in contributing! This document covers the three main ways to contribute:

1. [Adding an adapter](#adding-an-adapter)
2. [Proposing tier changes](#proposing-tier-changes)
3. [Dev setup](#dev-setup)

---

## Adding an Adapter

An adapter wraps any data validation tool so DQBench can benchmark it. The interface is intentionally minimal — one class, one method.

### Step 1: Create the adapter file

Create `dqbench/adapters/<toolname>.py`:

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
        import mytool
        return mytool.__version__

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        # Run your tool and translate its output into DQBenchFindings
        results = mytool.scan(csv_path)
        findings = []
        for issue in results.issues:
            findings.append(
                DQBenchFinding(
                    column=issue.column_name,
                    severity="error",   # "error", "warning", or "info"
                    check=issue.check_type,
                    message=issue.description,
                    confidence=issue.score,  # optional float 0.0-1.0
                )
            )
        return findings
```

### Step 2: Register it (if it's a built-in)

Add an entry to `dqbench/adapters/__init__.py` so it's discoverable by name:

```python
BUILTIN_ADAPTERS = {
    "goldencheck": "dqbench.adapters.goldencheck:GoldenCheckAdapter",
    "mytool": "dqbench.adapters.mytool:MyToolAdapter",
}
```

Add the optional dependency to `pyproject.toml`:

```toml
[project.optional-dependencies]
mytool = ["mytool>=1.0"]
```

### Step 3: Add tests

Create `tests/test_adapter_mytool.py` with at least a smoke test:

```python
pytest.importorskip("mytool")

from dqbench.adapters.mytool import MyToolAdapter
from dqbench.models import DQBenchFinding
from pathlib import Path


def test_adapter_returns_findings(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("email\nbad-email\ngood@example.com\n")
    adapter = MyToolAdapter()
    findings = adapter.validate(csv)
    assert isinstance(findings, list)
    assert all(isinstance(f, DQBenchFinding) for f in findings)
```

### Step 4: Open a PR

- Title: `feat: add <toolname> adapter`
- Include a link to the tool's repository
- The maintainer will run the full benchmark and post scores in the PR

---

## Proposing Tier Changes

The three tiers are designed to be stable and comparable across versions. Changes to tier content affect historical comparability, so they are treated carefully.

### What can change freely

- Bug fixes in generators (wrong dtype, broken seed, etc.)
- Documentation improvements
- Adding new issue *types* to adversarial (Tier 3) as a minor version bump

### What requires a discussion first

- Adding or removing columns from any tier
- Changing row counts
- Changing the scoring weights (20% / 40% / 40%)
- Any change that would alter scores for existing tools

To propose a structural tier change:

1. Open a GitHub Discussion under the **Tier Proposals** category
2. Include: motivation, proposed change, expected score impact
3. Tag it `tier-proposal`
4. Allow at least two weeks for community feedback before a PR is opened

### Versioning policy

Benchmark versions are immutable once published. A structural tier change results in a new benchmark version (e.g., `dqbench-v2`), not a modification of the existing one. This preserves historical comparability.

---

## Dev Setup

### Requirements

- Python 3.11+
- git

### Install in editable mode with dev dependencies

```bash
git clone https://github.com/benzsevern/dqbench
cd dqbench
pip install -e ".[dev]"
```

### Run tests

```bash
pytest --tb=short -v
```

### Lint

```bash
ruff check .
```

### Run the full benchmark locally

```bash
# With the built-in GoldenCheck adapter (requires goldencheck)
pip install -e ".[goldencheck]"
dqbench run goldencheck

# With a custom adapter file
dqbench run --adapter path/to/my_adapter.py
```

### Project layout

```
dqbench/
  adapters/      # Built-in adapters (goldencheck, etc.)
  generator/     # Tier dataset generators (tier1, tier2, tier3)
  cli.py         # Typer CLI entrypoint
  ground_truth.py # Ground truth column registry
  models.py      # DQBenchFinding, DQBenchResult, etc.
  runner.py      # Orchestrates adapter + scorer
  scorer.py      # Recall, precision, F1, DQBench Score
  report.py      # Rich scorecard output
tests/
  test_tier1_generator.py
  test_tier2_generator.py
  test_tier3_generator.py
  test_scorer.py
  test_runner.py
  ...
```

### Code style

- Line length: 100 (enforced by ruff)
- No type: ignore comments without a brief explanation
- New public functions need docstrings

---

Questions? Open an issue or start a Discussion.
