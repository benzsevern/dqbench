# DQBench: Entity Resolution & Pipeline Benchmark Categories

**Date:** 2026-03-29
**Status:** Approved

## Context

DQBench currently benchmarks two categories: **Detect** (data validation tools) and **Transform** (data cleaning tools). The Golden Suite includes two additional tools — GoldenMatch (entity resolution) and GoldenPipe (pipeline orchestration) — that have no standardized benchmarks. Adding ER and Pipeline categories to DQBench gives every Golden Suite tool a DQBench score and establishes DQBench as a comprehensive data quality benchmark.

## Architecture: Parallel Category Model

Each new category follows the same pattern as Detect and Transform: its own adapter interface, generator, ground truth model, scorer, and runner function. Categories are independent — adding ER does not affect existing Detect/Transform scores.

```
Category        Adapter Interface            Score Name
──────────────────────────────────────────────────────────
Detect          DQBenchAdapter.validate()     DQBench Detect: 88.40
Transform       TransformAdapter.transform()  DQBench Transform: 100.00
ER (new)        ERAdapter.deduplicate()       DQBench ER: XX.XX
Pipeline (new)  PipelineAdapter.run()         DQBench Pipeline: XX.XX
```

All categories use the same 20/40/40 tier weighting for composite scores.

---

## 1. Adapter Interfaces

Added to `adapters/base.py`. All adapters share a common base for `name`/`version`:

```python
class BenchmarkAdapter(ABC):
    """Shared base for all adapter types."""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...


class DQBenchAdapter(BenchmarkAdapter):       # existing, re-parented
    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: ...

class TransformAdapter(BenchmarkAdapter):      # existing, re-parented
    @abstractmethod
    def transform(self, csv_path: Path) -> pl.DataFrame: ...

class EntityResolutionAdapter(BenchmarkAdapter):  # NEW
    @abstractmethod
    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        """Return list of (row_a, row_b) matched pairs. 0-based row indices."""
        ...

class PipelineAdapter(BenchmarkAdapter):          # NEW
    @abstractmethod
    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        """Run full pipeline (validate → transform → deduplicate).
        Return the final cleaned, deduplicated DataFrame."""
        ...
```

ER returns **matched pairs** (universal — every ER tool can produce pairs). Pipeline returns the **final DataFrame** (compared against clean+deduplicated ground truth). Existing adapters are re-parented under `BenchmarkAdapter` — no interface change, just DRY.

---

## 2. ER Data Generation

### Synthetic Tiers

| Tier | Unique Entities | Dupe Pairs | Total Rows | Dupe Rate | Difficulty |
|------|----------------|------------|------------|-----------|------------|
| 1 | 900 | 100 | 1,000 | 10% | Easy |
| 2 | 4,250 | 750 | 5,000 | 15% | Fuzzy |
| 3 | 8,000 | 2,000 | 10,000 | 20% | Adversarial |

Each duplicate pair adds one row to the dataset (a variant of an existing entity). Dupe rate = duplicate_rows / total_rows. Domain: customer contacts.

**Columns:** first_name, last_name, email, phone, address, city, state, zip, company.

**Tier 1 — Easy (100 pairs):**
- 50 exact duplicate pairs (same values, different casing)
- 30 near-exact pairs (single typo in name or email)
- 20 pairs with swapped first/last name

**Tier 2 — Fuzzy:**
- Nickname variants ("Robert" / "Bob", "Elizabeth" / "Liz")
- Transposed fields (city in address field)
- Missing data (one record has phone, the other doesn't)
- Format differences ("(555) 123-4567" vs "5551234567")
- Mixed-case, extra whitespace, abbreviated titles

**Tier 3 — Adversarial:**
- Phonetic matches ("Smith" / "Smyth", "Thompson" / "Thomson")
- Abbreviations ("St." / "Street", "Jr." / "Junior")
- Merged records (two people at same address — false positive traps)
- Split records (one person's data across two rows with different info)
- Unicode confusables, zero-width characters in names

### Real Datasets (Tier 4, Optional)

Downloaded on demand via `dqbench generate --real`:

- **DBLP-ACM** — bibliographic matching (~2,600 records)
- **Abt-Buy** — product matching (~1,000 records)

Ground truth mappings bundled. Real dataset scores reported separately, not part of the composite DQBench ER score.

### ER Ground Truth Model

Pydantic `BaseModel`, consistent with existing `GroundTruth` in `ground_truth.py`:

```python
class ERGroundTruth(BaseModel):
    tier: int
    version: str
    rows: int
    duplicate_pairs: list[tuple[int, int]]  # true match pairs
    total_duplicates: int
    difficulty: str  # "easy", "fuzzy", "adversarial"
```

Cached at `~/.dqbench/datasets/er_tier{1,2,3}/` with `data.csv` and `er_ground_truth.json`. Real datasets cached at `~/.dqbench/datasets/er_real_{name}/`.

Loader follows existing pattern:
```python
def load_er_ground_truth(path: Path) -> ERGroundTruth:
    with open(path) as f:
        return ERGroundTruth(**json.load(f))
```

---

## 3. ER Scoring

Pair-level precision, recall, F1. Symmetric matching: `(a, b)` matches `(b, a)`.

```python
@dataclass
class ERTierResult:
    tier: int
    precision: float      # correct_pairs / predicted_pairs
    recall: float         # correct_pairs / true_pairs
    f1: float             # harmonic mean
    false_positives: int
    false_negatives: int
    time_seconds: float
    memory_mb: float

@dataclass
class ERScorecard:
    tool_name: str
    tool_version: str
    tiers: list[ERTierResult]
    real_datasets: list[ERRealResult] | None

    @property
    def dqbench_er_score(self) -> float:
        """T1 F1 × 0.20 + T2 F1 × 0.40 + T3 F1 × 0.40, scaled to 0-100."""
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.f1 * weights[t.tier] * 100 for t in self.tiers), 2)
```

```python
@dataclass
class ERRealResult:
    dataset_name: str
    precision: float
    recall: float
    f1: float
    time_seconds: float
```

Transitivity is NOT assumed — if the tool outputs clusters, they must be decomposed into all pairwise combinations before scoring.

---

## 4. Pipeline Benchmark

### Data Generation

Each pipeline tier combines quality issues AND duplicates in one dataset. The generator:
1. Generates clean base data
2. Plants duplicate rows (using ER tier logic)
3. Plants quality issues on top (using detect/transform tier logic)

This produces datasets that require a pipeline tool to fix issues AND resolve duplicates.

### Pipeline Ground Truth

Pydantic `BaseModel`. The clean+deduplicated DataFrame is stored as a separate CSV file, NOT embedded in the model (DataFrames are not JSON-serializable).

```python
class PipelineGroundTruth(BaseModel):
    tier: int
    version: str
    rows: int                              # rows in messy input
    planted_issues: int                    # quality issues planted
    duplicate_pairs: list[tuple[int, int]] # true match pairs
    expected_output_rows: int              # rows after deduplication
    # NOTE: clean+deduped DataFrame stored separately as data_clean_deduped.csv
```

**Cached files per tier:**
- `data.csv` — messy input with issues + duplicates
- `data_clean_deduped.csv` — ideal final output (clean values, duplicates removed)
- `pipeline_ground_truth.json` — metadata (no DataFrame field)

Loader:
```python
def load_pipeline_ground_truth(path: Path) -> PipelineGroundTruth:
    with open(path) as f:
        return PipelineGroundTruth(**json.load(f))

def load_pipeline_clean_df(tier_dir: Path) -> pl.DataFrame:
    return pl.read_csv(tier_dir / "data_clean_deduped.csv", infer_schema_length=0)
```

### Pipeline Scoring

Two dimensions, weighted into a composite. ER inference is dropped — instead we compare the output DataFrame against the clean+deduplicated ground truth holistically.

**Scoring algorithm:**
1. **Transform accuracy:** For rows that survive in the output, compare cell values against the corresponding rows in `data_clean_deduped.csv`. Alignment is by a deterministic `_row_id` column planted in the data (hidden from the tool, preserved through transforms). `accuracy = correct_cells / total_planted_cells`.
2. **Dedup accuracy:** `1.0 - abs(output_rows - expected_rows) / expected_rows`, clamped to [0, 1]. Perfect score when the tool produces exactly the expected row count.

Composite: `transform_accuracy × 0.6 + dedup_accuracy × 0.4`.

```python
@dataclass
class PipelineTierResult:
    tier: int
    transform_accuracy: float  # cell-level vs clean ground truth (0-1)
    dedup_accuracy: float      # row count accuracy (0-1)
    composite: float           # transform × 0.6 + dedup × 0.4
    output_rows: int
    expected_rows: int
    time_seconds: float
    memory_mb: float

@dataclass
class PipelineScorecard:
    tool_name: str
    tool_version: str
    tiers: list[PipelineTierResult]

    @property
    def dqbench_pipeline_score(self) -> float:
        """T1 × 0.20 + T2 × 0.40 + T3 × 0.40, scaled to 0-100."""
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.composite * weights[t.tier] * 100 for t in self.tiers), 2)
```

**Row alignment strategy:** The generator plants a `_row_id` column in the data. This is a monotonically increasing integer that survives transforms but allows the scorer to match output rows to ground truth rows. After deduplication, the scorer joins on `_row_id` to compare cell values. Rows with `_row_id` values from duplicate rows that were correctly removed are expected to be absent.

---

## 5. CLI Integration

### New Commands

```bash
# ER benchmarks
dqbench run goldenmatch              # synthetic tiers 1-3
dqbench run goldenmatch --real       # also run real datasets
dqbench run goldenmatch --tier 2     # single tier

# Pipeline benchmarks
dqbench run goldenpipe               # tiers 1-3

# Dataset generation
dqbench generate --er                # ER synthetic datasets
dqbench generate --pipeline          # pipeline datasets
dqbench generate --real              # download real ER datasets
dqbench generate --all               # everything

# Category comparisons
dqbench run all --er                 # all ER adapters
dqbench run all --pipeline           # all pipeline adapters
```

### Adapter Auto-Detection and CLI Dispatch

The `_load_adapter()` function in `cli.py` is updated to detect all four types via `isinstance` checks:

```python
def _detect_category(adapter) -> str:
    if isinstance(adapter, PipelineAdapter):
        return "pipeline"
    if isinstance(adapter, EntityResolutionAdapter):
        return "er"
    if isinstance(adapter, TransformAdapter):
        return "transform"
    return "detect"  # DQBenchAdapter default
```

The `run` command dispatches to the correct runner:

```python
category = _detect_category(adapter)
if category == "er":
    scorecard = run_er_benchmark(adapter, tiers=tiers, real=real)
    report_er(scorecard, json_output=json_flag)
elif category == "pipeline":
    scorecard = run_pipeline_benchmark(adapter, tiers=tiers)
    report_pipeline(scorecard, json_output=json_flag)
elif category == "transform":
    scorecard = run_transform_benchmark(adapter, tiers=tiers)
    report_transform(scorecard, json_output=json_flag)
else:
    scorecard = run_benchmark(adapter, tiers=tiers)
    report_detect(scorecard, json_output=json_flag)
```

### `run all` with Category Filtering

The `all` subcommand gains `--er` and `--pipeline` flags. These are **filtering flags for `run all` only**, not used with individual adapter names.

```python
# Adapter lists by category
DETECT_ADAPTERS = ["goldencheck", "gx-zero", "gx-auto", ...]
TRANSFORM_ADAPTERS = ["goldenflow", ...]
ER_ADAPTERS = ["goldenmatch"]
PIPELINE_ADAPTERS = ["goldenpipe"]

def _run_all(category: str = "detect"):
    adapters = {
        "detect": DETECT_ADAPTERS,
        "transform": TRANSFORM_ADAPTERS,
        "er": ER_ADAPTERS,
        "pipeline": PIPELINE_ADAPTERS,
    }[category]
    # ... run each and produce comparison table
```

All commands support `--json` for structured output, consistent with existing behavior.

### Adapter Registry

```python
BUILTIN_ADAPTERS = {
    # existing detect adapters...
    # existing transform adapters...
    "goldenmatch": "dqbench.adapters.goldenmatch_adapter:GoldenMatchAdapter",
    "goldenpipe": "dqbench.adapters.goldenpipe_adapter:GoldenPipeAdapter",
}
```

---

## 6. File Structure

### New Files (~16)

```
dqbench/
├── adapters/
│   ├── base.py                        # MODIFY: add 2 new ABCs
│   ├── goldenmatch_adapter.py         # NEW
│   └── goldenpipe_adapter.py          # NEW
├── generator/
│   ├── er_tier1.py                    # NEW: 1K rows, easy dupes
│   ├── er_tier2.py                    # NEW: 5K rows, fuzzy dupes
│   ├── er_tier3.py                    # NEW: 10K rows, adversarial
│   ├── er_real.py                     # NEW: DBLP-ACM, Abt-Buy download
│   ├── pipeline_tier1.py             # NEW: issues + easy dupes
│   ├── pipeline_tier2.py             # NEW: issues + fuzzy dupes
│   └── pipeline_tier3.py             # NEW: issues + adversarial dupes
├── er_ground_truth.py                 # NEW
├── er_scorer.py                       # NEW
├── pipeline_ground_truth.py           # NEW
├── pipeline_scorer.py                 # NEW
├── models.py                          # MODIFY: add ER + Pipeline dataclasses
├── runner.py                          # MODIFY: add 2 new runner functions
├── cli.py                             # MODIFY: new adapters + flags
└── report.py                          # MODIFY: ER + Pipeline report tables
```

### New Test Files (~4)

```
tests/
├── test_er_generator.py               # NEW
├── test_er_scorer.py                  # NEW
├── test_pipeline_generator.py         # NEW
└── test_pipeline_scorer.py            # NEW
```

---

## 7. Constraints

- **Determinism:** All generators use `random.Random(42)`, no numpy
- **No external deps:** Only polars, typer, rich, pydantic (goldenmatch/goldenpipe are optional for adapters)
- **Immutability:** Published tier datasets never change
- **Backward compat:** Existing Detect/Transform scores unaffected
- **Cache dir:** `~/.dqbench/datasets/` — new dirs prefixed with `er_` and `pipeline_`
- **Real dataset downloads:** If download fails (network error, dataset unavailable), print a warning and skip gracefully — do not fail the benchmark run. Cache a marker file so subsequent runs know to retry.
- **Row ID column:** Pipeline datasets include a `_row_id` column for scorer alignment. Adapters should preserve this column (or drop it — the scorer handles both cases by falling back to positional matching).

## 8. Verification

- `dqbench generate --all` creates all datasets without error
- `dqbench run goldenmatch` produces an ERScorecard with valid F1 scores
- `dqbench run goldenpipe` produces a PipelineScorecard with composite scores
- `dqbench run goldenmatch --real` downloads and scores real datasets
- All existing `dqbench run goldencheck` / `dqbench run goldenflow` scores unchanged
- Tests pass: `pytest tests/test_er_*.py tests/test_pipeline_*.py`
