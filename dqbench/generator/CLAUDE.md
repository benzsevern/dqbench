# Generator

Produces deterministic benchmark datasets for each tier.

## Tiers

| Tier | Rows | Cols | Theme |
|------|------|------|-------|
| 1 | 5,000 | 20 | Customer database — basics |
| 2 | 50,000 | 30 | Transactions — realistic noise |
| 3 | 100,000 | 50 | Healthcare/finance — adversarial |

Each tier has **planted columns** (with deliberate issues) and **clean columns**
(false-positive traps — well-formed data that must not be flagged).

## Return Type

Every generator returns `(pl.DataFrame, GroundTruth)`.

```python
df, gt = generate_tier1()
df.write_csv("data.csv")
gt.model_dump()  # serialise to JSON
```

## Determinism Rule

Use a local `rng = random.Random(42)` instance passed through all helpers.
Never call `random.seed(42)` globally — that mutates shared state and breaks
tests that run in the same process.

## Ground Truth Format

`GroundTruth.planted_columns` is a `dict[str, PlantedColumn]` where each entry is:

```python
PlantedColumn(
    issues=["null_values", "invalid_format"],  # issue type keys
    planted_count=50,                           # how many rows affected
    description="email col with nulls + bad formats",
    affected_rows=[3, 7, 12, ...],             # optional row indices
)
```

`GroundTruth.clean_columns` is a plain `list[str]` — columns with no planted issues.

## Adding a New Tier

1. Create `dqbench/generator/tier4.py` following the existing pattern.
2. Return `(pl.DataFrame, GroundTruth(tier=4, version="1.0", ...))`.
3. Register in `runner.py` `ensure_datasets()` loop.
4. Add a weight entry in `Scorecard.dqbench_score` if it should affect the composite.

## Modifying an Existing Tier

Ground truth versions are **immutable once published**. If you change planted
issues, bump `version` in the returned `GroundTruth` and document the change.
Old cached files at `~/.dqbench/datasets/` will not auto-invalidate — users
must run `dqbench generate --force`.
