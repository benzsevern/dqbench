# DQBench

The standard benchmark for data quality and validation tools.

## Commands

```bash
pip install -e ".[dev]"          # Dev install
pytest --tb=short -v             # Run tests (83 passing)
ruff check .                     # Lint
dqbench run <adapter>            # Run benchmark
dqbench run all                  # Head-to-head comparison
dqbench generate                 # Generate/cache datasets
dqbench generate --force         # Regenerate from scratch
```

## Architecture

```
dqbench/
├── cli.py           # Typer CLI (run, generate, results)
├── runner.py        # Orchestrate adapter against tiers
├── scorer.py        # Compute recall, precision, F1, DQBench Score
├── report.py        # Rich console + JSON scorecard
├── models.py        # DQBenchFinding, TierResult, Scorecard
├── ground_truth.py  # GroundTruth Pydantic model + loader
├── generator/       # Tier 1/2/3 dataset generators
└── adapters/        # Tool adapters (base ABC + built-ins)
```

## Scoring

- **Column Recall**: any finding on planted column (any severity) = detected
- **Column FPR**: WARNING/ERROR on clean column = false positive (INFO is NOT FP)
- **Issue Recall**: finding must match planted issue TYPE (via keyword matching)
- **Issue Precision**: matched findings / total findings on planted columns + FPs
- **DQBench Score**: `T1_issue_F1 × 20% + T2_issue_F1 × 40% + T3_issue_F1 × 40%` (0-100)

## Key Patterns

- **Datasets are deterministic**: `random.Random(42)` (stdlib only, no numpy)
- **Datasets cached**: `~/.dqbench/datasets/` — regenerate with `--force`
- **Issue matching uses keywords**: `ISSUE_KEYWORDS` dict in `scorer.py`
- **Adapter interface**: one class, one method (`validate(csv_path) -> list[DQBenchFinding]`)
- **Three modes per tool**: zero-config, auto-profiled, best-effort

## Gotchas

- Ground truth versions are immutable once published
- `ISSUE_KEYWORDS` must not be changed after benchmark is locked — tools tune to it
- Tier generators use `random.Random(42)` instance, not global `random.seed(42)`
- GitHub auth: `gh auth switch --user benzsevern` before pushing
