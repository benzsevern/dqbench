# Transform Benchmarks for dqbench — Design Spec

**Date:** 2026-03-25
**Status:** Approved
**Scope:** Add data transformation benchmarking to dqbench alongside existing detection benchmarks

---

## Overview

Extend dqbench to benchmark **data transformation tools** (e.g., GoldenFlow) in addition to data quality detection tools (e.g., GoldenCheck). Transformation tools take messy data and produce clean data. dqbench measures how accurately they do it by comparing output against ground truth clean CSVs.

Reuses the existing 3-tier dataset infrastructure. Adds clean ground truth CSVs, a `TransformAdapter` interface, cell-level accuracy scoring, and a GoldenFlow adapter as the first transform tool.

---

## Adapter Interface

New abstract base class alongside the existing `DQBenchAdapter`:

```python
class TransformAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def transform(self, csv_path: Path) -> pl.DataFrame:
        """Transform the messy CSV and return the cleaned DataFrame."""
        ...
```

Minimal interface: one method, one return type. dqbench owns all scoring logic by diffing the returned DataFrame against the ground truth clean CSV.

The runner detects adapter type (`DQBenchAdapter` vs `TransformAdapter`) and routes to the appropriate scoring pipeline.

---

## Ground Truth Clean CSVs

For each existing tier, generate a `_clean.csv` alongside the messy data:

- `tier1_clean.csv` — 5,000 rows, customer database domain
- `tier2_clean.csv` — 50,000 rows, e-commerce domain
- `tier3_clean.csv` — 100,000 rows, healthcare claims domain

### Generation Strategy

The generator is restructured to a **clean-first, then mutate** approach:

1. Generate the full clean dataset using `random.seed(42)` — all values are correct
2. Write out `_clean.csv`
3. Copy the clean data and apply mutations (planted issues) to create the messy version
4. Write out `data.csv`

This ensures every planted-issue cell has a known-correct original value. The clean CSV is generated in the same `ensure_datasets` pipeline alongside the messy data and ground truth JSON.

### Transformable vs Detection-Only Issues

Not all planted issues have a deterministic "correct" transformation. Issues are categorized:

**Transformable** (scored for transform tools):
- Format issues: phone formats → E.164, date formats → ISO 8601, zip codes → zero-padded 5-digit
- Case/whitespace: emails → lowercase + stripped, names → proper-cased + stripped
- Encoding: Unicode normalization
- Categorical: status misspellings → canonical, null variants → null, country codes → canonical
- Type coercion: numeric strings stored as text → correct type

**Detection-only** (excluded from transform scoring):
- Duplicates (no single correct resolution — dedup is GoldenMatch's job)
- Outliers (no ground truth for what the value "should" be)
- Logic violations (e.g., login before signup — ambiguous which date to fix)
- Correlated nulls (no original value exists to restore)
- Planted nulls in required fields (no original value exists)

The ground truth JSON is extended with a `"transformable": true/false` flag per planted issue. Only transformable issues contribute to the transform score.

### Per-Tier Clean Value Rules

**Tier 1 (customer database):**
- `phone` → E.164 (`+1XXXXXXXXXX`)
- `email` → lowercase, stripped
- `signup_date`, `last_login` → ISO 8601
- `state` → two-letter abbreviation
- `zip` → zero-padded 5-digit
- `status` → canonical spelling ("Active", "Inactive", "Pending")
- `country` → ISO 3166-1 alpha-2

**Tier 2 (e-commerce):**
- `customer_email` → lowercase, stripped
- `order_date`, `ship_date` → ISO 8601
- `zip_code` → zero-padded 5-digit
- `product_category` → canonical spelling
- `order_total` → numeric (strip currency symbols)

**Tier 3 (healthcare claims):**
- `patient_email` → lowercase, stripped
- `service_date`, `submit_date` → ISO 8601
- `patient_zip` → zero-padded 5-digit
- `icd10_code` → uppercase, properly formatted (with dot)
- `npi_number` → 10-digit zero-padded string

Issue types not listed above (duplicates, outliers, logic violations, correlated nulls, planted nulls) are detection-only and excluded from transform scoring.

---

## Scoring

### Cell-Level Accuracy

For each tier, score only cells that differ between messy and clean ground truth (i.e., cells with planted issues):

```
accuracy = correct_cells / total_planted_cells
```

Where:
- `correct_cells` = tool output matches clean ground truth (exact string match after strip)
- `total_planted_cells` = cells that differ between messy and clean CSVs

Cells that didn't need transformation are excluded from scoring — they don't inflate or deflate the score.

### Per-Column Accuracy

For each column with planted issues:
```
column_accuracy = correct_cells_in_column / planted_cells_in_column
```

Reported in the detailed breakdown.

### Composite Score

Same 0-100 scale and tier weighting as detection:

```
Transform Score = (T1_accuracy × 20%) + (T2_accuracy × 40%) + (T3_accuracy × 40%)
```

Directly comparable to detection scores. A tool scoring 80 on transformation is meaningfully comparable to a tool scoring 80 on detection.

### What Counts as Correct

- Cell matches ground truth exactly (string comparison after stripping both sides)
- Cells the tool chose not to transform → scored as incorrect (missed fix)
- Cells the tool transformed to the wrong value → scored as incorrect
- Cells the tool correctly transformed → scored as correct
- Null handling: tool output `null`/`None` matches ground truth `null`/`None`

### Output Shape Requirements

The returned DataFrame must have:
- **Same number of rows** in the same order as the input CSV
- **Same column names** as the input CSV (columns may not be added, removed, or renamed)
- If a tool violates these constraints, the tier scores 0 and the report notes the violation

### TransformTierResult Model

```python
@dataclass
class TransformColumnResult:
    column: str
    planted_cells: int
    correct_cells: int
    wrong_cells: int
    accuracy: float

@dataclass
class TransformTierResult:
    tier: int
    accuracy: float              # correct / total planted
    correct_cells: int
    wrong_cells: int
    planted_cells: int
    time_seconds: float
    memory_mb: float
    per_column: list[TransformColumnResult]

@dataclass
class TransformScorecard:
    tool_name: str
    tool_version: str
    tiers: list[TransformTierResult]
    composite_score: float       # weighted 20/40/40
```

### Runner Routing

`run_benchmark` uses `isinstance` to detect adapter type and calls the appropriate pipeline:

```python
if isinstance(adapter, TransformAdapter):
    result_df = adapter.transform(csv_path)
    tier_result = score_transform_tier(result_df, clean_df, ground_truth, tier)
elif isinstance(adapter, DQBenchAdapter):
    findings = adapter.validate(csv_path)
    tier_result = score_tier(findings, ground_truth, tier)
```

This is a targeted refactor of the existing `run_benchmark` function — the detection path remains unchanged.

### Definition of "Skipped"

"Skipped" = planted cells where the tool's output is identical to the messy input (i.e., the tool did not attempt to transform the cell). Skipped cells are a subset of wrong cells — they count as incorrect. The "Skipped" column in the report is informational, showing how many planted issues the tool missed entirely vs. attempted but got wrong.

---

## CLI Integration

Transform tools use the same `dqbench run` command:

```bash
dqbench run goldenflow                    # transform benchmark (all tiers)
dqbench run goldenflow --tier 2           # single tier
dqbench run goldenflow --json             # JSON output
dqbench run all                           # all tools (detection + transform)
dqbench run all --json                    # comparison across all tools
```

The runner auto-detects adapter type via `isinstance` check (`TransformAdapter` vs `DQBenchAdapter`) and routes to the appropriate scoring pipeline.

### CLI Changes Required

- `cli.py`: Update `BUILTIN_ADAPTERS` dict and `ALL_ADAPTER_NAMES` list to include `"goldenflow"`. Update `_load_adapter` to check for both `validate` (detection) and `transform` (transform) methods when loading custom adapters.
- No new CLI commands needed — the `run` command handles both adapter types.

---

## Report Output

### Transform Report (Rich table)

```
                    Transform Benchmark: GoldenFlow v0.1.0
+------+----------+----------+----------+---------+---------+--------+
| Tier | Accuracy | Correct  | Wrong    | Skipped | Time    | Memory |
+------+----------+----------+----------+---------+---------+--------+
| T1   | 89.3%    | 2,145    | 256      | 0       | 0.42s   | 12 MB  |
| T2   | 82.1%    | 8,420    | 1,834    | 0       | 3.21s   | 45 MB  |
| T3   | 78.6%    | 15,230   | 4,150    | 0       | 8.90s   | 89 MB  |
+------+----------+----------+----------+---------+---------+--------+
| DQBench Transform Score: 81.2 / 100                                |
+------+----------+----------+----------+---------+---------+--------+
```

### Per-Column Breakdown

```
                    Tier 1: Per-Column Accuracy
+---------------+----------+----------+--------+
| Column        | Accuracy | Correct  | Planted|
+---------------+----------+----------+--------+
| phone         | 95.2%    | 476      | 500    |
| email         | 100.0%   | 300      | 300    |
| signup_date   | 88.5%    | 354      | 400    |
| state         | 72.0%    | 180      | 250    |
| ...           |          |          |        |
+---------------+----------+----------+--------+
```

### Comparison Report (when `dqbench run all`)

Detection and transform tools shown in separate sections with their respective scores, since the metrics are different (F1 vs accuracy) but on the same 0-100 scale.

---

## Files to Add/Modify

### New Files

```
dqbench/
├── adapters/
│   └── goldenflow.py        # GoldenFlow TransformAdapter
└── generator/
    └── clean.py             # Clean CSV generation for all tiers
```

### Modified Files

```
dqbench/
├── adapters/
│   ├── base.py              # Add TransformAdapter ABC
│   └── __init__.py          # Register goldenflow adapter
├── generator/
│   ├── tier1.py             # Export clean generation helper
│   ├── tier2.py             # Export clean generation helper
│   └── tier3.py             # Export clean generation helper
├── models.py                # Add TransformResult model
├── scorer.py                # Add transform scoring (cell-level accuracy)
├── runner.py                # Detect adapter type, route scoring
├── report.py                # Add transform report tables + comparison sections
└── cli.py                   # Update BUILTIN_ADAPTERS, ALL_ADAPTER_NAMES, _load_adapter
```

---

## GoldenFlow Adapter

First transform tool adapter:

```python
class GoldenFlowAdapter(TransformAdapter):
    @property
    def name(self) -> str:
        return "GoldenFlow"

    @property
    def version(self) -> str:
        try:
            import goldenflow
            return goldenflow.__version__
        except ImportError:
            return "not installed"

    def transform(self, csv_path: Path) -> pl.DataFrame:
        import goldenflow
        result = goldenflow.transform_file(csv_path)
        return result.df
```

Zero-config — GoldenFlow's auto-detection engine decides what to transform. This matches how GoldenCheck's adapter uses zero-config scanning.

---

## Dependencies

- `goldenflow` added as optional dependency: `transform = ["goldenflow>=0.1"]`
- No new core dependencies — reuses polars, rich, pydantic, typer

---

## Comparison Report Behavior

When `dqbench run all` includes both detection and transform tools:
- `report_comparison` produces **two separate tables**: "Detection Tools" and "Transform Tools"
- Each table uses its own metric (F1 for detection, accuracy for transform)
- Both show the composite 0-100 score for direct comparison
- The function signature is extended to accept both `list[Scorecard]` and `list[TransformScorecard]`

---

## Test Plan

### New Test Files
- `tests/test_transform_scorer.py` — unit tests for `score_transform_tier()`
- `tests/test_transform_adapter.py` — test GoldenFlow adapter (mock if not installed)

### Modified Test Files
- `tests/test_runner.py` — add tests for `isinstance` routing to transform pipeline
- `tests/test_cli.py` — add test that `goldenflow` appears in adapter list
- `tests/test_generator_tier1.py` — verify `_clean.csv` is generated alongside `data.csv`

### Key Test Scenarios
1. **Perfect transform**: tool output matches clean CSV exactly → accuracy 100%
2. **No-op transform**: tool returns input unchanged → accuracy 0% (all planted cells wrong)
3. **Partial transform**: tool fixes some issues, misses others → proportional accuracy
4. **Wrong shape**: tool returns different row count → tier scores 0 with error message
5. **Missing columns**: tool drops a column → tier scores 0 with error message

---

## What This Does NOT Include

- Schema mapping benchmarks (different problem, different ground truth format)
- Name splitting benchmarks (requires column-count changes, not cell-level comparison)
- Custom config benchmarks (only zero-config / auto-detect mode is scored)
- Cross-tool comparison scoring (detection F1 and transform accuracy are on the same scale but measure different things — reported separately)
