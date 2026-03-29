# DQBench ER & Pipeline Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Entity Resolution and Pipeline benchmark categories to DQBench, giving GoldenMatch and GoldenPipe standardized DQBench scores.

**Architecture:** Parallel category model — each new category gets its own adapter interface, generator, ground truth model, scorer, and runner function, following the same patterns as existing Detect and Transform categories.

**Tech Stack:** Python 3.11+, Polars, Pydantic, Typer, Rich

---

## Task 1: Adapter Base Classes

**Files:**
- **Modify:** `dqbench/adapters/base.py`
- **Test:** `tests/test_adapters_base.py`

### Steps

- [ ] **1.1** Write failing test `tests/test_adapters_base.py`:

```python
"""Tests for adapter base classes and inheritance."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import pytest

from dqbench.adapters.base import (
    BenchmarkAdapter,
    DQBenchAdapter,
    TransformAdapter,
    EntityResolutionAdapter,
    PipelineAdapter,
)
from dqbench.models import DQBenchFinding


class FakeDetectAdapter(DQBenchAdapter):
    @property
    def name(self) -> str:
        return "fake-detect"

    @property
    def version(self) -> str:
        return "0.1.0"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        return []


class FakeTransformAdapter(TransformAdapter):
    @property
    def name(self) -> str:
        return "fake-transform"

    @property
    def version(self) -> str:
        return "0.1.0"

    def transform(self, csv_path: Path) -> pl.DataFrame:
        return pl.DataFrame()


class FakeERAdapter(EntityResolutionAdapter):
    @property
    def name(self) -> str:
        return "fake-er"

    @property
    def version(self) -> str:
        return "0.1.0"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        return []


class FakePipelineAdapter(PipelineAdapter):
    @property
    def name(self) -> str:
        return "fake-pipeline"

    @property
    def version(self) -> str:
        return "0.1.0"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        return pl.DataFrame()


def test_all_adapters_inherit_benchmark_adapter():
    """All four adapter types share BenchmarkAdapter as their root."""
    detect = FakeDetectAdapter()
    transform = FakeTransformAdapter()
    er = FakeERAdapter()
    pipeline = FakePipelineAdapter()

    for adapter in [detect, transform, er, pipeline]:
        assert isinstance(adapter, BenchmarkAdapter)


def test_detect_adapter_still_works():
    adapter = FakeDetectAdapter()
    assert adapter.name == "fake-detect"
    assert adapter.version == "0.1.0"
    assert adapter.validate(Path("x.csv")) == []


def test_transform_adapter_still_works():
    adapter = FakeTransformAdapter()
    assert adapter.name == "fake-transform"
    assert adapter.transform(Path("x.csv")).shape == (0, 0)


def test_er_adapter_interface():
    adapter = FakeERAdapter()
    assert adapter.name == "fake-er"
    assert adapter.deduplicate(Path("x.csv")) == []


def test_pipeline_adapter_interface():
    adapter = FakePipelineAdapter()
    assert adapter.name == "fake-pipeline"
    assert adapter.run_pipeline(Path("x.csv")).shape == (0, 0)


def test_isinstance_discrimination():
    """Each adapter type is distinguishable via isinstance."""
    detect = FakeDetectAdapter()
    er = FakeERAdapter()
    pipeline = FakePipelineAdapter()

    assert isinstance(detect, DQBenchAdapter)
    assert not isinstance(detect, EntityResolutionAdapter)
    assert isinstance(er, EntityResolutionAdapter)
    assert not isinstance(er, PipelineAdapter)
    assert isinstance(pipeline, PipelineAdapter)
    assert not isinstance(pipeline, EntityResolutionAdapter)
```

- [ ] **1.2** Run test, confirm failure:
```bash
pytest tests/test_adapters_base.py -v
# Expected: ImportError — BenchmarkAdapter, EntityResolutionAdapter, PipelineAdapter do not exist
```

- [ ] **1.3** Modify `dqbench/adapters/base.py` to the following:

```python
"""Base adapter interfaces for all benchmark categories."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl
from dqbench.models import DQBenchFinding


class BenchmarkAdapter(ABC):
    """Shared base for all adapter types."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...


class DQBenchAdapter(BenchmarkAdapter):
    """Adapter for Detect category (data validation tools)."""

    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: ...


class TransformAdapter(BenchmarkAdapter):
    """Adapter for Transform category (data cleaning tools)."""

    @abstractmethod
    def transform(self, csv_path: Path) -> pl.DataFrame:
        """Transform the messy CSV and return the cleaned DataFrame."""
        ...


class EntityResolutionAdapter(BenchmarkAdapter):
    """Adapter for ER category (entity resolution / deduplication tools)."""

    @abstractmethod
    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        """Return list of (row_a, row_b) matched pairs. 0-based row indices."""
        ...


class PipelineAdapter(BenchmarkAdapter):
    """Adapter for Pipeline category (end-to-end data quality pipeline tools)."""

    @abstractmethod
    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        """Run full pipeline (validate -> transform -> deduplicate).
        Return the final cleaned, deduplicated DataFrame."""
        ...
```

- [ ] **1.4** Run test, confirm all pass:
```bash
pytest tests/test_adapters_base.py -v
# Expected: 6 passed
```

- [ ] **1.5** Verify existing adapters still import and instantiate (spot check):
```bash
python -c "from dqbench.adapters.base import DQBenchAdapter, TransformAdapter; print('OK')"
```

- [ ] **1.6** Commit:
```
feat(adapters): add BenchmarkAdapter base, EntityResolutionAdapter, PipelineAdapter

Re-parent DQBenchAdapter and TransformAdapter under shared BenchmarkAdapter ABC.
Add EntityResolutionAdapter.deduplicate() and PipelineAdapter.run_pipeline() interfaces.
```

---

## Task 2: ER and Pipeline Data Models

**Files:**
- **Modify:** `dqbench/models.py`
- **Test:** `tests/test_models_er_pipeline.py`

### Steps

- [ ] **2.1** Write failing test `tests/test_models_er_pipeline.py`:

```python
"""Tests for ER and Pipeline dataclass models."""
from __future__ import annotations
import pytest

from dqbench.models import (
    ERTierResult,
    ERScorecard,
    ERRealResult,
    PipelineTierResult,
    PipelineScorecard,
)


def test_er_tier_result_creation():
    r = ERTierResult(
        tier=1, precision=0.9, recall=0.8, f1=0.8471,
        false_positives=10, false_negatives=20,
        time_seconds=1.5, memory_mb=50.0,
    )
    assert r.tier == 1
    assert r.precision == 0.9
    assert r.f1 == 0.8471


def test_er_real_result_creation():
    r = ERRealResult(
        dataset_name="DBLP-ACM",
        precision=0.95, recall=0.88, f1=0.9136,
        time_seconds=3.2,
    )
    assert r.dataset_name == "DBLP-ACM"


def test_er_scorecard_composite_score():
    tiers = [
        ERTierResult(tier=1, precision=1.0, recall=1.0, f1=1.0,
                     false_positives=0, false_negatives=0,
                     time_seconds=0.1, memory_mb=10.0),
        ERTierResult(tier=2, precision=0.8, recall=0.8, f1=0.8,
                     false_positives=5, false_negatives=5,
                     time_seconds=0.5, memory_mb=20.0),
        ERTierResult(tier=3, precision=0.6, recall=0.6, f1=0.6,
                     false_positives=10, false_negatives=10,
                     time_seconds=1.0, memory_mb=30.0),
    ]
    sc = ERScorecard(tool_name="test", tool_version="1.0", tiers=tiers, real_datasets=None)
    # 1.0*0.20*100 + 0.8*0.40*100 + 0.6*0.40*100 = 20 + 32 + 24 = 76.0
    assert sc.dqbench_er_score == 76.0


def test_er_scorecard_with_real_datasets():
    tiers = [
        ERTierResult(tier=1, precision=1.0, recall=1.0, f1=1.0,
                     false_positives=0, false_negatives=0,
                     time_seconds=0.1, memory_mb=10.0),
        ERTierResult(tier=2, precision=1.0, recall=1.0, f1=1.0,
                     false_positives=0, false_negatives=0,
                     time_seconds=0.1, memory_mb=10.0),
        ERTierResult(tier=3, precision=1.0, recall=1.0, f1=1.0,
                     false_positives=0, false_negatives=0,
                     time_seconds=0.1, memory_mb=10.0),
    ]
    real = [ERRealResult(dataset_name="DBLP-ACM", precision=0.9, recall=0.85, f1=0.874, time_seconds=2.0)]
    sc = ERScorecard(tool_name="test", tool_version="1.0", tiers=tiers, real_datasets=real)
    assert sc.dqbench_er_score == 100.0
    assert len(sc.real_datasets) == 1


def test_pipeline_tier_result_creation():
    r = PipelineTierResult(
        tier=1, transform_accuracy=0.95, dedup_accuracy=1.0,
        composite=0.95 * 0.6 + 1.0 * 0.4,
        output_rows=900, expected_rows=900,
        time_seconds=2.0, memory_mb=60.0,
    )
    assert r.tier == 1
    assert r.composite == pytest.approx(0.97)


def test_pipeline_scorecard_composite_score():
    tiers = [
        PipelineTierResult(tier=1, transform_accuracy=1.0, dedup_accuracy=1.0,
                           composite=1.0, output_rows=900, expected_rows=900,
                           time_seconds=0.1, memory_mb=10.0),
        PipelineTierResult(tier=2, transform_accuracy=0.8, dedup_accuracy=0.9,
                           composite=0.8*0.6 + 0.9*0.4,
                           output_rows=4000, expected_rows=4250,
                           time_seconds=0.5, memory_mb=20.0),
        PipelineTierResult(tier=3, transform_accuracy=0.6, dedup_accuracy=0.7,
                           composite=0.6*0.6 + 0.7*0.4,
                           output_rows=7500, expected_rows=8000,
                           time_seconds=1.0, memory_mb=30.0),
    ]
    sc = PipelineScorecard(tool_name="test", tool_version="1.0", tiers=tiers)
    # T1: 1.0*0.20*100=20, T2: 0.84*0.40*100=33.6, T3: 0.64*0.40*100=25.6
    assert sc.dqbench_pipeline_score == 79.2
```

- [ ] **2.2** Run test, confirm failure:
```bash
pytest tests/test_models_er_pipeline.py -v
# Expected: ImportError — ERTierResult etc. do not exist
```

- [ ] **2.3** Add the following dataclasses to the end of `dqbench/models.py`:

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
class ERRealResult:
    dataset_name: str
    precision: float
    recall: float
    f1: float
    time_seconds: float


@dataclass
class ERScorecard:
    tool_name: str
    tool_version: str
    tiers: list[ERTierResult]
    real_datasets: list[ERRealResult] | None

    @property
    def dqbench_er_score(self) -> float:
        """T1 F1 x 0.20 + T2 F1 x 0.40 + T3 F1 x 0.40, scaled to 0-100."""
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.f1 * weights[t.tier] * 100 for t in self.tiers), 2)


@dataclass
class PipelineTierResult:
    tier: int
    transform_accuracy: float  # cell-level vs clean ground truth (0-1)
    dedup_accuracy: float      # row count accuracy (0-1)
    composite: float           # transform x 0.6 + dedup x 0.4
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
        """T1 x 0.20 + T2 x 0.40 + T3 x 0.40, scaled to 0-100."""
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.composite * weights[t.tier] * 100 for t in self.tiers), 2)
```

- [ ] **2.4** Run test, confirm all pass:
```bash
pytest tests/test_models_er_pipeline.py -v
# Expected: 7 passed
```

- [ ] **2.5** Commit:
```
feat(models): add ERTierResult, ERScorecard, ERRealResult, PipelineTierResult, PipelineScorecard

Dataclass models for ER and Pipeline benchmark categories, following
existing TierResult/Scorecard pattern with 20/40/40 tier weighting.
```

---

## Task 3: ER Ground Truth

**Files:**
- **Create:** `dqbench/er_ground_truth.py`
- **Test:** `tests/test_er_ground_truth.py`

### Steps

- [ ] **3.1** Write failing test `tests/test_er_ground_truth.py`:

```python
"""Tests for ER ground truth model and loader."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

from dqbench.er_ground_truth import ERGroundTruth, load_er_ground_truth


def test_er_ground_truth_creation():
    gt = ERGroundTruth(
        tier=1,
        version="1.0.0",
        rows=1000,
        duplicate_pairs=[(0, 900), (1, 901), (2, 902)],
        total_duplicates=3,
        difficulty="easy",
    )
    assert gt.tier == 1
    assert gt.rows == 1000
    assert len(gt.duplicate_pairs) == 3
    assert gt.difficulty == "easy"


def test_er_ground_truth_roundtrip():
    """Serialize to JSON and deserialize back — values must match."""
    gt = ERGroundTruth(
        tier=2,
        version="1.0.0",
        rows=5000,
        duplicate_pairs=[(10, 4300), (55, 4400)],
        total_duplicates=2,
        difficulty="fuzzy",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt.model_dump(), f)
        path = Path(f.name)

    loaded = load_er_ground_truth(path)
    assert loaded.tier == gt.tier
    assert loaded.version == gt.version
    assert loaded.rows == gt.rows
    assert loaded.duplicate_pairs == gt.duplicate_pairs
    assert loaded.total_duplicates == gt.total_duplicates
    assert loaded.difficulty == gt.difficulty
    path.unlink()


def test_er_ground_truth_pairs_are_tuples():
    """Pairs stored as lists in JSON should be converted to tuples."""
    data = {
        "tier": 1,
        "version": "1.0.0",
        "rows": 100,
        "duplicate_pairs": [[0, 50], [1, 51]],
        "total_duplicates": 2,
        "difficulty": "easy",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    loaded = load_er_ground_truth(path)
    for pair in loaded.duplicate_pairs:
        assert isinstance(pair, tuple)
    path.unlink()
```

- [ ] **3.2** Run test, confirm failure:
```bash
pytest tests/test_er_ground_truth.py -v
# Expected: ModuleNotFoundError — dqbench.er_ground_truth does not exist
```

- [ ] **3.3** Create `dqbench/er_ground_truth.py`:

```python
"""Load and query ER ground truth manifests."""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel


class ERGroundTruth(BaseModel):
    tier: int
    version: str
    rows: int
    duplicate_pairs: list[tuple[int, int]]
    total_duplicates: int
    difficulty: str  # "easy", "fuzzy", "adversarial"


def load_er_ground_truth(path: Path) -> ERGroundTruth:
    with open(path) as f:
        return ERGroundTruth(**json.load(f))
```

- [ ] **3.4** Run test, confirm all pass:
```bash
pytest tests/test_er_ground_truth.py -v
# Expected: 3 passed
```

- [ ] **3.5** Commit:
```
feat(er): add ERGroundTruth Pydantic model and loader

Follows existing GroundTruth pattern in ground_truth.py.
Stores duplicate pairs as list[tuple[int, int]] with JSON round-trip support.
```

---

## Task 4: ER Tier 1 Generator (Easy Dupes)

**Files:**
- **Create:** `dqbench/generator/er_tier1.py`
- **Create:** `dqbench/generator/er_utils.py` (shared ER helper functions: `_generate_phone`, `_generate_address`, `_generate_entity` — used by all ER and Pipeline tier generators to avoid duplication)
- **Modify:** `dqbench/generator/utils.py` (add ER-specific data pools)
- **Test:** `tests/test_er_generator.py`

> **DRY Note:** Extract `_generate_phone`, `_generate_address`, and `_generate_entity` into `er_utils.py`. All ER and Pipeline tier generators import from here instead of duplicating these helpers.

### Steps

- [ ] **4.1** Add ER-specific data pools to `dqbench/generator/utils.py`:

```python
# ---- ER data pools ----

COMPANIES = [
    "Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries",
    "Wayne Enterprises", "Cyberdyne", "Soylent Corp", "Oscorp", "LexCorp",
    "Wonka Industries", "Aperture Science", "Tyrell Corp", "Weyland-Yutani",
    "Massive Dynamic", "Hooli", "Pied Piper", "Dunder Mifflin", "Sterling Cooper",
    "Prestige Worldwide",
]

STREET_NAMES = [
    "Main St", "Oak Ave", "Elm St", "Park Blvd", "Maple Dr",
    "Cedar Ln", "Pine Rd", "Washington St", "Lake Ave", "Hill St",
    "River Rd", "Spring St", "Forest Dr", "Sunset Blvd", "Highland Ave",
    "Broadway", "Market St", "Church St", "Mill Rd", "Center St",
]

PHONE_AREA_CODES = [
    "212", "310", "312", "415", "512", "617", "702", "713", "718", "773",
    "818", "917", "202", "305", "404", "503", "602", "614", "704", "916",
]
```

- [ ] **4.2** Write failing test `tests/test_er_generator.py`:

```python
"""Tests for ER tier generators."""
from __future__ import annotations
import polars as pl
import pytest

from dqbench.generator.er_tier1 import generate_er_tier1
from dqbench.er_ground_truth import ERGroundTruth


class TestERTier1:
    def test_returns_dataframe_and_ground_truth(self):
        df, gt = generate_er_tier1()
        assert isinstance(df, pl.DataFrame)
        assert isinstance(gt, ERGroundTruth)

    def test_row_count(self):
        df, gt = generate_er_tier1()
        assert df.shape[0] == 1000
        assert gt.rows == 1000

    def test_expected_columns(self):
        df, _ = generate_er_tier1()
        expected = {"first_name", "last_name", "email", "phone",
                    "address", "city", "state", "zip", "company"}
        assert set(df.columns) == expected

    def test_duplicate_pair_count(self):
        _, gt = generate_er_tier1()
        assert gt.total_duplicates == 100
        assert len(gt.duplicate_pairs) == 100

    def test_duplicate_pair_breakdown(self):
        """50 case-change, 30 typo, 20 name-swap = 100 total."""
        _, gt = generate_er_tier1()
        assert len(gt.duplicate_pairs) == 100

    def test_pairs_reference_valid_rows(self):
        df, gt = generate_er_tier1()
        n = df.shape[0]
        for a, b in gt.duplicate_pairs:
            assert 0 <= a < n, f"Invalid row index {a}"
            assert 0 <= b < n, f"Invalid row index {b}"
            assert a != b, f"Self-pair ({a}, {a})"

    def test_determinism(self):
        """Two calls produce identical output."""
        df1, gt1 = generate_er_tier1()
        df2, gt2 = generate_er_tier1()
        assert df1.frame_equal(df2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, gt = generate_er_tier1()
        assert gt.tier == 1
        assert gt.difficulty == "easy"
        assert gt.version == "1.0.0"
```

- [ ] **4.3** Run test, confirm failure:
```bash
pytest tests/test_er_generator.py::TestERTier1 -v
# Expected: ModuleNotFoundError — dqbench.generator.er_tier1 does not exist
```

- [ ] **4.4** Create `dqbench/generator/er_tier1.py`:

```python
"""ER Tier 1 dataset generator — 1,000 rows with 100 easy duplicate pairs."""
from __future__ import annotations

import random
from typing import Any

import polars as pl

from dqbench.er_ground_truth import ERGroundTruth
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COMPANIES,
    STREET_NAMES,
    PHONE_AREA_CODES,
)

NROWS = 1000
N_UNIQUE = 900
N_DUPES = 100  # 50 case-change + 30 typo + 20 name-swap


def _generate_phone(rng: random.Random) -> str:
    area = rng.choice(PHONE_AREA_CODES)
    return f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _generate_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"


def _generate_entity(rng: random.Random) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zipcode = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    email = f"{first.lower()}.{last.lower()}@{domain}"
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": _generate_phone(rng),
        "address": _generate_address(rng),
        "city": city,
        "state": state,
        "zip": zipcode,
        "company": rng.choice(COMPANIES),
    }


def _case_change_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Create a duplicate with case changes."""
    dupe = entity.copy()
    if rng.random() < 0.5:
        dupe["first_name"] = dupe["first_name"].upper()
    else:
        dupe["first_name"] = dupe["first_name"].lower()
    if rng.random() < 0.5:
        dupe["last_name"] = dupe["last_name"].upper()
    else:
        dupe["last_name"] = dupe["last_name"].lower()
    dupe["email"] = dupe["email"].upper()
    return dupe


def _typo_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Create a duplicate with a single typo in name or email."""
    dupe = entity.copy()
    field = rng.choice(["first_name", "last_name", "email"])
    val = list(dupe[field])
    if len(val) > 2:
        idx = rng.randint(1, len(val) - 2)
        # Swap two adjacent characters
        val[idx], val[idx - 1] = val[idx - 1], val[idx]
        dupe[field] = "".join(val)
    return dupe


def _name_swap_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Create a duplicate with first/last name swapped."""
    dupe = entity.copy()
    dupe["first_name"], dupe["last_name"] = entity["last_name"], entity["first_name"]
    return dupe


def generate_er_tier1() -> tuple[pl.DataFrame, ERGroundTruth]:
    """Generate the ER Tier 1 benchmark dataset."""
    rng = random.Random(42)

    # Generate 900 unique entities
    entities: list[dict[str, str]] = []
    for _ in range(N_UNIQUE):
        entities.append(_generate_entity(rng))

    # Select 100 source entities to duplicate
    source_indices = rng.sample(range(N_UNIQUE), N_DUPES)

    # Split: 50 case-change, 30 typo, 20 name-swap
    case_sources = source_indices[:50]
    typo_sources = source_indices[50:80]
    swap_sources = source_indices[80:100]

    duplicate_pairs: list[tuple[int, int]] = []
    dupe_rows: list[dict[str, str]] = []

    for src_idx in case_sources:
        dupe_row_idx = N_UNIQUE + len(dupe_rows)
        dupe_rows.append(_case_change_dupe(entities[src_idx], rng))
        duplicate_pairs.append((src_idx, dupe_row_idx))

    for src_idx in typo_sources:
        dupe_row_idx = N_UNIQUE + len(dupe_rows)
        dupe_rows.append(_typo_dupe(entities[src_idx], rng))
        duplicate_pairs.append((src_idx, dupe_row_idx))

    for src_idx in swap_sources:
        dupe_row_idx = N_UNIQUE + len(dupe_rows)
        dupe_rows.append(_name_swap_dupe(entities[src_idx], rng))
        duplicate_pairs.append((src_idx, dupe_row_idx))

    # Combine and shuffle
    all_rows = entities + dupe_rows
    assert len(all_rows) == NROWS

    # Shuffle rows, tracking index remapping
    indices = list(range(NROWS))
    rng.shuffle(indices)
    shuffled_rows = [all_rows[i] for i in indices]

    # Remap duplicate pairs to shuffled positions
    old_to_new = {old: new for new, old in enumerate(indices)}
    remapped_pairs = [
        (min(old_to_new[a], old_to_new[b]), max(old_to_new[a], old_to_new[b]))
        for a, b in duplicate_pairs
    ]
    remapped_pairs.sort()

    df = pl.DataFrame(shuffled_rows)

    gt = ERGroundTruth(
        tier=1,
        version="1.0.0",
        rows=NROWS,
        duplicate_pairs=remapped_pairs,
        total_duplicates=N_DUPES,
        difficulty="easy",
    )

    return df, gt
```

- [ ] **4.5** Run test, confirm all pass:
```bash
pytest tests/test_er_generator.py::TestERTier1 -v
# Expected: 8 passed
```

- [ ] **4.6** Commit:
```
feat(er): add ER Tier 1 generator — 1000 rows with 100 easy duplicate pairs

50 case-change, 30 typo, 20 name-swap duplicates.
Deterministic via random.Random(42). Reuses shared data pools from utils.py.
```

---

## Task 5: ER Scorer

**Files:**
- **Create:** `dqbench/er_scorer.py`
- **Test:** `tests/test_er_scorer.py`

### Steps

- [ ] **5.1** Write failing test `tests/test_er_scorer.py`:

```python
"""Tests for ER scorer."""
from __future__ import annotations
import pytest

from dqbench.er_scorer import score_er_tier
from dqbench.er_ground_truth import ERGroundTruth
from dqbench.models import ERTierResult


@pytest.fixture
def ground_truth() -> ERGroundTruth:
    return ERGroundTruth(
        tier=1,
        version="1.0.0",
        rows=100,
        duplicate_pairs=[(0, 50), (1, 51), (2, 52), (3, 53), (4, 54)],
        total_duplicates=5,
        difficulty="easy",
    )


class TestScoreERTier:
    def test_perfect_predictions(self, ground_truth):
        """All true pairs predicted, no false positives."""
        predictions = [(0, 50), (1, 51), (2, 52), (3, 53), (4, 54)]
        result = score_er_tier(predictions, ground_truth, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        assert isinstance(result, ERTierResult)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_symmetric_matching(self, ground_truth):
        """(a, b) should match (b, a)."""
        predictions = [(50, 0), (51, 1), (52, 2), (53, 3), (54, 4)]
        result = score_er_tier(predictions, ground_truth, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_partial_predictions(self, ground_truth):
        """Only 3 of 5 true pairs predicted."""
        predictions = [(0, 50), (1, 51), (2, 52)]
        result = score_er_tier(predictions, ground_truth, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        assert result.precision == 1.0
        assert result.recall == pytest.approx(0.6)
        assert result.false_positives == 0
        assert result.false_negatives == 2

    def test_with_false_positives(self, ground_truth):
        """All true pairs plus 2 false positives."""
        predictions = [(0, 50), (1, 51), (2, 52), (3, 53), (4, 54),
                        (10, 60), (20, 70)]
        result = score_er_tier(predictions, ground_truth, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        assert result.recall == 1.0
        assert result.precision == pytest.approx(5 / 7)
        assert result.false_positives == 2
        assert result.false_negatives == 0

    def test_empty_predictions(self, ground_truth):
        """No predictions at all."""
        result = score_er_tier([], ground_truth, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.false_negatives == 5

    def test_no_true_pairs(self):
        """Ground truth has no pairs, predictions are all false positives."""
        gt = ERGroundTruth(
            tier=1, version="1.0.0", rows=100,
            duplicate_pairs=[], total_duplicates=0, difficulty="easy",
        )
        predictions = [(0, 1), (2, 3)]
        result = score_er_tier(predictions, gt, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        assert result.precision == 0.0
        assert result.recall == 0.0  # no true pairs => recall defined as 0
        assert result.false_positives == 2

    def test_f1_harmonic_mean(self, ground_truth):
        """F1 should be harmonic mean of precision and recall."""
        predictions = [(0, 50), (1, 51), (10, 60)]  # 2 correct, 1 FP
        result = score_er_tier(predictions, ground_truth, tier=1,
                               time_seconds=1.0, memory_mb=10.0)
        p = 2 / 3
        r = 2 / 5
        expected_f1 = 2 * p * r / (p + r)
        assert result.f1 == pytest.approx(expected_f1)
```

- [ ] **5.2** Run test, confirm failure:
```bash
pytest tests/test_er_scorer.py -v
# Expected: ModuleNotFoundError
```

- [ ] **5.3** Create `dqbench/er_scorer.py`:

```python
"""Scoring logic for ER benchmarks."""
from __future__ import annotations

from dqbench.models import ERTierResult
from dqbench.er_ground_truth import ERGroundTruth


def _normalize_pairs(pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Normalize pairs to (min, max) for symmetric matching."""
    return {(min(a, b), max(a, b)) for a, b in pairs}


def score_er_tier(
    predictions: list[tuple[int, int]],
    ground_truth: ERGroundTruth,
    tier: int,
    time_seconds: float,
    memory_mb: float,
) -> ERTierResult:
    """Score ER predictions against ground truth using pair-level P/R/F1."""
    true_pairs = _normalize_pairs(ground_truth.duplicate_pairs)
    pred_pairs = _normalize_pairs(predictions)

    true_positives = pred_pairs & true_pairs
    false_positives = pred_pairs - true_pairs
    false_negatives = true_pairs - pred_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    if tp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    return ERTierResult(
        tier=tier,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=fp,
        false_negatives=fn,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
    )
```

- [ ] **5.4** Run test, confirm all pass:
```bash
pytest tests/test_er_scorer.py -v
# Expected: 7 passed
```

- [ ] **5.5** Commit:
```
feat(er): add ER scorer with pair-level precision/recall/F1

Symmetric matching — (a,b) matches (b,a) via normalization to (min,max).
Handles edge cases: empty predictions, no true pairs, partial matches.
```

---

## Task 6: ER Runner

**Files:**
- **Modify:** `dqbench/runner.py`
- **Test:** `tests/test_er_runner.py`

### Steps

- [ ] **6.1** Write failing test `tests/test_er_runner.py`:

```python
"""Tests for ER runner integration."""
from __future__ import annotations
import shutil
from pathlib import Path

import polars as pl
import pytest

from dqbench.adapters.base import EntityResolutionAdapter
from dqbench.models import ERScorecard


class PerfectERAdapter(EntityResolutionAdapter):
    """Returns exact ground truth pairs for testing."""

    def __init__(self, pairs: list[tuple[int, int]] | None = None):
        self._pairs = pairs or []

    @property
    def name(self) -> str:
        return "perfect-er"

    @property
    def version(self) -> str:
        return "0.0.1"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        if self._pairs:
            return self._pairs
        # Load ground truth to return perfect predictions
        import json
        gt_path = csv_path.parent / "er_ground_truth.json"
        with open(gt_path) as f:
            data = json.load(f)
        return [tuple(p) for p in data["duplicate_pairs"]]


class TestERRunner:
    def test_run_er_benchmark_returns_scorecard(self):
        from dqbench.runner import run_er_benchmark, ensure_er_datasets
        ensure_er_datasets()
        adapter = PerfectERAdapter()
        scorecard = run_er_benchmark(adapter, tiers=[1])
        assert isinstance(scorecard, ERScorecard)
        assert scorecard.tool_name == "perfect-er"
        assert len(scorecard.tiers) == 1

    def test_perfect_er_scores_100(self):
        from dqbench.runner import run_er_benchmark, ensure_er_datasets
        ensure_er_datasets()
        adapter = PerfectERAdapter()
        scorecard = run_er_benchmark(adapter, tiers=[1])
        assert scorecard.tiers[0].f1 == 1.0
        assert scorecard.tiers[0].precision == 1.0
        assert scorecard.tiers[0].recall == 1.0

    def test_ensure_er_datasets_creates_files(self):
        from dqbench.runner import CACHE_DIR, ensure_er_datasets
        ensure_er_datasets()
        assert (CACHE_DIR / "er_tier1" / "data.csv").exists()
        assert (CACHE_DIR / "er_tier1" / "er_ground_truth.json").exists()
```

- [ ] **6.2** Run test, confirm failure:
```bash
pytest tests/test_er_runner.py -v
# Expected: ImportError — run_er_benchmark, ensure_er_datasets do not exist
```

- [ ] **6.3** Add `ensure_er_datasets()` and `run_er_benchmark()` to `dqbench/runner.py`. Add the following imports at the top of the file:

```python
from dqbench.adapters.base import DQBenchAdapter, TransformAdapter, EntityResolutionAdapter
from dqbench.models import Scorecard, TransformScorecard, ERScorecard
```

Then add the following functions after `run_transform_benchmark()`:

```python
def ensure_er_datasets() -> None:
    """Generate ER datasets if not cached."""
    if (CACHE_DIR / "er_tier1" / "data.csv").exists():
        return
    from dqbench.generator.er_tier1 import generate_er_tier1

    generators = [(1, generate_er_tier1)]

    # Import tier 2 and 3 if available
    try:
        from dqbench.generator.er_tier2 import generate_er_tier2
        generators.append((2, generate_er_tier2))
    except ImportError:
        pass
    try:
        from dqbench.generator.er_tier3 import generate_er_tier3
        generators.append((3, generate_er_tier3))
    except ImportError:
        pass

    for tier_num, gen_fn in generators:
        tier_dir = CACHE_DIR / f"er_tier{tier_num}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        df, gt = gen_fn()
        df.write_csv(tier_dir / "data.csv")
        with open(tier_dir / "er_ground_truth.json", "w") as f:
            json.dump(gt.model_dump(), f, indent=2)


def run_er_benchmark(
    adapter: EntityResolutionAdapter,
    tiers: list[int] | None = None,
    real: bool = False,
) -> ERScorecard:
    """Run ER benchmark and return a scorecard."""
    from dqbench.er_scorer import score_er_tier
    from dqbench.er_ground_truth import load_er_ground_truth

    ensure_er_datasets()
    tier_nums = tiers or [1, 2, 3]
    results = []

    for tier_num in tier_nums:
        tier_dir = CACHE_DIR / f"er_tier{tier_num}"
        csv_path = tier_dir / "data.csv"
        gt_path = tier_dir / "er_ground_truth.json"

        if not csv_path.exists():
            continue

        gt = load_er_ground_truth(gt_path)

        tracemalloc.start()
        t0 = time.perf_counter()
        predictions = adapter.deduplicate(csv_path)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = score_er_tier(
            predictions, gt, tier=tier_num,
            time_seconds=elapsed, memory_mb=peak / (1024 * 1024),
        )
        results.append(result)

    return ERScorecard(
        tool_name=adapter.name,
        tool_version=adapter.version,
        tiers=results,
        real_datasets=None,
    )
```

- [ ] **6.4** Run test, confirm all pass:
```bash
pytest tests/test_er_runner.py -v
# Expected: 3 passed
```

- [ ] **6.5** Commit:
```
feat(er): add ensure_er_datasets() and run_er_benchmark() to runner

Follows existing run_benchmark/run_transform_benchmark pattern.
Generates ER tier datasets on first run, caches in ~/.dqbench/datasets/er_tier{1,2,3}/.
```

---

## Task 7: ER Tier 2 and Tier 3 Generators

**Files:**
- **Create:** `dqbench/generator/er_tier2.py`
- **Create:** `dqbench/generator/er_tier3.py`
- **Modify:** `dqbench/generator/utils.py` (add nickname mappings, phonetic variants)
- **Test:** `tests/test_er_generator.py` (add TestERTier2 and TestERTier3)

### Steps

- [ ] **7.1** Add Tier 2 and 3 data pools to `dqbench/generator/utils.py`:

```python
# ---- ER Tier 2: Nickname mappings ----
NICKNAME_MAP: dict[str, list[str]] = {
    "Robert": ["Bob", "Rob", "Bobby"],
    "Elizabeth": ["Liz", "Beth", "Lizzy"],
    "William": ["Bill", "Will", "Billy"],
    "James": ["Jim", "Jimmy"],
    "Richard": ["Rick", "Dick", "Rich"],
    "Michael": ["Mike", "Mikey"],
    "Jennifer": ["Jen", "Jenny"],
    "Patricia": ["Pat", "Patty"],
    "Margaret": ["Meg", "Maggie", "Peggy"],
    "Joseph": ["Joe", "Joey"],
    "Thomas": ["Tom", "Tommy"],
    "Christopher": ["Chris"],
    "Daniel": ["Dan", "Danny"],
    "Matthew": ["Matt"],
    "Anthony": ["Tony"],
    "Steven": ["Steve"],
    "Kenneth": ["Ken", "Kenny"],
    "Timothy": ["Tim", "Timmy"],
    "Jessica": ["Jess", "Jessie"],
    "Barbara": ["Barb"],
}

# ---- ER Tier 3: Phonetic variants ----
PHONETIC_VARIANTS: dict[str, list[str]] = {
    "Smith": ["Smyth", "Smithe"],
    "Thompson": ["Thomson", "Tompson"],
    "Johnson": ["Johnsen", "Jonson"],
    "Williams": ["Willams", "Wiliams"],
    "Anderson": ["Andersen", "Andersson"],
    "Martinez": ["Martines", "Martinz"],
    "Wilson": ["Willson", "Wilsen"],
    "Taylor": ["Tailor", "Tayler"],
    "Moore": ["More", "Moor"],
    "Jackson": ["Jacksen", "Jaxon"],
}

ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "St": "Street",
    "Ave": "Avenue",
    "Blvd": "Boulevard",
    "Dr": "Drive",
    "Ln": "Lane",
    "Rd": "Road",
}
```

- [ ] **7.2** Add test classes for Tier 2 and 3 to `tests/test_er_generator.py`:

```python
from dqbench.generator.er_tier2 import generate_er_tier2
from dqbench.generator.er_tier3 import generate_er_tier3


class TestERTier2:
    def test_returns_dataframe_and_ground_truth(self):
        df, gt = generate_er_tier2()
        assert isinstance(df, pl.DataFrame)
        assert isinstance(gt, ERGroundTruth)

    def test_row_count(self):
        df, gt = generate_er_tier2()
        assert df.shape[0] == 5000
        assert gt.rows == 5000

    def test_expected_columns(self):
        df, _ = generate_er_tier2()
        expected = {"first_name", "last_name", "email", "phone",
                    "address", "city", "state", "zip", "company"}
        assert set(df.columns) == expected

    def test_duplicate_pair_count(self):
        _, gt = generate_er_tier2()
        assert gt.total_duplicates == 750
        assert len(gt.duplicate_pairs) == 750

    def test_determinism(self):
        df1, gt1 = generate_er_tier2()
        df2, gt2 = generate_er_tier2()
        assert df1.frame_equal(df2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, gt = generate_er_tier2()
        assert gt.tier == 2
        assert gt.difficulty == "fuzzy"


class TestERTier3:
    def test_returns_dataframe_and_ground_truth(self):
        df, gt = generate_er_tier3()
        assert isinstance(df, pl.DataFrame)
        assert isinstance(gt, ERGroundTruth)

    def test_row_count(self):
        df, gt = generate_er_tier3()
        assert df.shape[0] == 10000
        assert gt.rows == 10000

    def test_expected_columns(self):
        df, _ = generate_er_tier3()
        expected = {"first_name", "last_name", "email", "phone",
                    "address", "city", "state", "zip", "company"}
        assert set(df.columns) == expected

    def test_duplicate_pair_count(self):
        _, gt = generate_er_tier3()
        assert gt.total_duplicates == 2000
        assert len(gt.duplicate_pairs) == 2000

    def test_determinism(self):
        df1, gt1 = generate_er_tier3()
        df2, gt2 = generate_er_tier3()
        assert df1.frame_equal(df2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, gt = generate_er_tier3()
        assert gt.tier == 3
        assert gt.difficulty == "adversarial"
```

- [ ] **7.3** Run tests, confirm failure:
```bash
pytest tests/test_er_generator.py::TestERTier2 tests/test_er_generator.py::TestERTier3 -v
# Expected: ImportError
```

- [ ] **7.4** Create `dqbench/generator/er_tier2.py`:

```python
"""ER Tier 2 dataset generator — 5,000 rows with 750 fuzzy duplicate pairs."""
from __future__ import annotations

import random

import polars as pl

from dqbench.er_ground_truth import ERGroundTruth
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COMPANIES,
    STREET_NAMES,
    PHONE_AREA_CODES,
    NICKNAME_MAP,
)

NROWS = 5000
N_UNIQUE = 4250
N_DUPES = 750


def _generate_phone(rng: random.Random) -> str:
    area = rng.choice(PHONE_AREA_CODES)
    return f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _generate_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"


def _generate_entity(rng: random.Random) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zipcode = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    email = f"{first.lower()}.{last.lower()}@{domain}"
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": _generate_phone(rng),
        "address": _generate_address(rng),
        "city": city,
        "state": state,
        "zip": zipcode,
        "company": rng.choice(COMPANIES),
    }


def _nickname_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Use a nickname variant for first_name."""
    dupe = entity.copy()
    first = dupe["first_name"]
    if first in NICKNAME_MAP:
        dupe["first_name"] = rng.choice(NICKNAME_MAP[first])
    else:
        dupe["first_name"] = first.lower()
    return dupe


def _missing_field_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Remove one field (set to empty string)."""
    dupe = entity.copy()
    field = rng.choice(["phone", "email", "address", "company"])
    dupe[field] = ""
    return dupe


def _format_change_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Change phone format and add whitespace."""
    dupe = entity.copy()
    # Strip phone formatting
    phone = dupe["phone"].replace("(", "").replace(")", "").replace(" ", "").replace("-", "")
    dupe["phone"] = phone
    # Add extra whitespace to name
    dupe["first_name"] = f"  {dupe['first_name']}  "
    return dupe


def _case_typo_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Mix of case change and typo."""
    dupe = entity.copy()
    dupe["first_name"] = dupe["first_name"].upper()
    dupe["last_name"] = dupe["last_name"].lower()
    # Typo in email
    email = list(dupe["email"])
    if len(email) > 4:
        idx = rng.randint(1, len(email) - 3)
        email[idx], email[idx + 1] = email[idx + 1], email[idx]
        dupe["email"] = "".join(email)
    return dupe


def _transposed_field_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """City value placed in address field."""
    dupe = entity.copy()
    dupe["address"] = dupe["city"]
    return dupe


def generate_er_tier2() -> tuple[pl.DataFrame, ERGroundTruth]:
    """Generate the ER Tier 2 benchmark dataset."""
    rng = random.Random(42)

    entities: list[dict[str, str]] = []
    for _ in range(N_UNIQUE):
        entities.append(_generate_entity(rng))

    source_indices = rng.sample(range(N_UNIQUE), N_DUPES)

    # Split across dupe strategies: 200 nickname, 150 missing, 150 format, 150 case+typo, 100 transposed
    dupe_fns = (
        [_nickname_dupe] * 200
        + [_missing_field_dupe] * 150
        + [_format_change_dupe] * 150
        + [_case_typo_dupe] * 150
        + [_transposed_field_dupe] * 100
    )

    duplicate_pairs: list[tuple[int, int]] = []
    dupe_rows: list[dict[str, str]] = []

    for i, src_idx in enumerate(source_indices):
        dupe_row_idx = N_UNIQUE + len(dupe_rows)
        dupe_rows.append(dupe_fns[i](entities[src_idx], rng))
        duplicate_pairs.append((src_idx, dupe_row_idx))

    all_rows = entities + dupe_rows
    assert len(all_rows) == NROWS

    indices = list(range(NROWS))
    rng.shuffle(indices)
    shuffled_rows = [all_rows[i] for i in indices]

    old_to_new = {old: new for new, old in enumerate(indices)}
    remapped_pairs = [
        (min(old_to_new[a], old_to_new[b]), max(old_to_new[a], old_to_new[b]))
        for a, b in duplicate_pairs
    ]
    remapped_pairs.sort()

    df = pl.DataFrame(shuffled_rows)

    gt = ERGroundTruth(
        tier=2,
        version="1.0.0",
        rows=NROWS,
        duplicate_pairs=remapped_pairs,
        total_duplicates=N_DUPES,
        difficulty="fuzzy",
    )

    return df, gt
```

- [ ] **7.5** Create `dqbench/generator/er_tier3.py`:

```python
"""ER Tier 3 dataset generator — 10,000 rows with 2,000 adversarial duplicate pairs."""
from __future__ import annotations

import random

import polars as pl

from dqbench.er_ground_truth import ERGroundTruth
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COMPANIES,
    STREET_NAMES,
    PHONE_AREA_CODES,
    NICKNAME_MAP,
    PHONETIC_VARIANTS,
    ADDRESS_ABBREVIATIONS,
)

NROWS = 10000
N_UNIQUE = 8000
N_DUPES = 2000


def _generate_phone(rng: random.Random) -> str:
    area = rng.choice(PHONE_AREA_CODES)
    return f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _generate_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"


def _generate_entity(rng: random.Random) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zipcode = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    email = f"{first.lower()}.{last.lower()}@{domain}"
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": _generate_phone(rng),
        "address": _generate_address(rng),
        "city": city,
        "state": state,
        "zip": zipcode,
        "company": rng.choice(COMPANIES),
    }


def _phonetic_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Use a phonetic variant for last_name."""
    dupe = entity.copy()
    last = dupe["last_name"]
    if last in PHONETIC_VARIANTS:
        dupe["last_name"] = rng.choice(PHONETIC_VARIANTS[last])
    else:
        # Insert a silent letter
        name = list(dupe["last_name"])
        idx = rng.randint(1, max(1, len(name) - 1))
        name.insert(idx, rng.choice(["h", "e"]))
        dupe["last_name"] = "".join(name)
    return dupe


def _abbreviation_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Expand or abbreviate address parts."""
    dupe = entity.copy()
    addr = dupe["address"]
    for abbr, full in ADDRESS_ABBREVIATIONS.items():
        if addr.endswith(abbr):
            dupe["address"] = addr[: -len(abbr)] + full
            break
        elif addr.endswith(full):
            dupe["address"] = addr[: -len(full)] + abbr
            break
    return dupe


def _split_record_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """One person's data split — keep name/email but blank other fields."""
    dupe = entity.copy()
    for field in ["phone", "address", "city", "state", "zip"]:
        if rng.random() < 0.6:
            dupe[field] = ""
    return dupe


def _unicode_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Insert zero-width characters or use unicode confusables."""
    dupe = entity.copy()
    # Insert zero-width space in first_name
    name = dupe["first_name"]
    if len(name) > 2:
        idx = rng.randint(1, len(name) - 1)
        dupe["first_name"] = name[:idx] + "\u200b" + name[idx:]
    return dupe


def _merged_record_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Same address but slightly different name — false positive trap if naive."""
    dupe = entity.copy()
    # Change first name to a different one to create a "roommate" record
    # This is a TRUE duplicate (same person, different name variant)
    if dupe["first_name"] in NICKNAME_MAP:
        dupe["first_name"] = rng.choice(NICKNAME_MAP[dupe["first_name"]])
    else:
        dupe["first_name"] = dupe["first_name"][::-1].capitalize()
    # Keep same address, company, etc.
    return dupe


def _multi_field_dupe(entity: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Multiple fields corrupted simultaneously."""
    dupe = entity.copy()
    dupe["first_name"] = dupe["first_name"].lower()
    dupe["last_name"] = dupe["last_name"].upper()
    # Reformat phone
    phone = dupe["phone"].replace("(", "").replace(")", "").replace(" ", "").replace("-", "")
    dupe["phone"] = phone
    # Typo in email
    email = list(dupe["email"])
    if len(email) > 3:
        idx = rng.randint(0, len(email) - 2)
        email[idx] = rng.choice("abcdefghijklmnop")
        dupe["email"] = "".join(email)
    return dupe


def generate_er_tier3() -> tuple[pl.DataFrame, ERGroundTruth]:
    """Generate the ER Tier 3 benchmark dataset."""
    rng = random.Random(42)

    entities: list[dict[str, str]] = []
    for _ in range(N_UNIQUE):
        entities.append(_generate_entity(rng))

    source_indices = rng.sample(range(N_UNIQUE), N_DUPES)

    # Split: 400 phonetic, 300 abbreviation, 300 split, 300 unicode,
    #        300 merged, 400 multi-field
    dupe_fns = (
        [_phonetic_dupe] * 400
        + [_abbreviation_dupe] * 300
        + [_split_record_dupe] * 300
        + [_unicode_dupe] * 300
        + [_merged_record_dupe] * 300
        + [_multi_field_dupe] * 400
    )

    duplicate_pairs: list[tuple[int, int]] = []
    dupe_rows: list[dict[str, str]] = []

    for i, src_idx in enumerate(source_indices):
        dupe_row_idx = N_UNIQUE + len(dupe_rows)
        dupe_rows.append(dupe_fns[i](entities[src_idx], rng))
        duplicate_pairs.append((src_idx, dupe_row_idx))

    all_rows = entities + dupe_rows
    assert len(all_rows) == NROWS

    indices = list(range(NROWS))
    rng.shuffle(indices)
    shuffled_rows = [all_rows[i] for i in indices]

    old_to_new = {old: new for new, old in enumerate(indices)}
    remapped_pairs = [
        (min(old_to_new[a], old_to_new[b]), max(old_to_new[a], old_to_new[b]))
        for a, b in duplicate_pairs
    ]
    remapped_pairs.sort()

    df = pl.DataFrame(shuffled_rows)

    gt = ERGroundTruth(
        tier=3,
        version="1.0.0",
        rows=NROWS,
        duplicate_pairs=remapped_pairs,
        total_duplicates=N_DUPES,
        difficulty="adversarial",
    )

    return df, gt
```

- [ ] **7.6** Run tests, confirm all pass:
```bash
pytest tests/test_er_generator.py -v
# Expected: 20 passed (8 tier1 + 6 tier2 + 6 tier3)
```

- [ ] **7.7** Commit:
```
feat(er): add ER Tier 2 (fuzzy) and Tier 3 (adversarial) generators

Tier 2: 5000 rows, 750 dupes — nicknames, missing fields, format changes,
transposed fields, mixed case+typo.
Tier 3: 10000 rows, 2000 dupes — phonetic variants, abbreviations, split
records, unicode confusables, merged records, multi-field corruption.
```

---

## Task 8: Pipeline Ground Truth, Generator, Scorer

**Files:**
- **Create:** `dqbench/pipeline_ground_truth.py`
- **Create:** `dqbench/generator/pipeline_tier1.py`
- **Create:** `dqbench/pipeline_scorer.py`
- **Test:** `tests/test_pipeline_ground_truth.py`
- **Test:** `tests/test_pipeline_scorer.py`

### Steps

- [ ] **8.1** Write failing test `tests/test_pipeline_ground_truth.py`:

```python
"""Tests for Pipeline ground truth model and loaders."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import polars as pl

from dqbench.pipeline_ground_truth import (
    PipelineGroundTruth,
    load_pipeline_ground_truth,
    load_pipeline_clean_df,
)


def test_pipeline_ground_truth_creation():
    gt = PipelineGroundTruth(
        tier=1,
        version="1.0.0",
        rows=1000,
        planted_issues=150,
        duplicate_pairs=[(0, 900), (1, 901)],
        expected_output_rows=900,
    )
    assert gt.tier == 1
    assert gt.planted_issues == 150
    assert gt.expected_output_rows == 900


def test_pipeline_ground_truth_roundtrip():
    gt = PipelineGroundTruth(
        tier=1,
        version="1.0.0",
        rows=1000,
        planted_issues=100,
        duplicate_pairs=[(5, 950)],
        expected_output_rows=999,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt.model_dump(), f)
        path = Path(f.name)

    loaded = load_pipeline_ground_truth(path)
    assert loaded.tier == gt.tier
    assert loaded.rows == gt.rows
    assert loaded.duplicate_pairs == gt.duplicate_pairs
    path.unlink()


def test_load_pipeline_clean_df():
    with tempfile.TemporaryDirectory() as tmpdir:
        tier_dir = Path(tmpdir)
        df = pl.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
        df.write_csv(tier_dir / "data_clean_deduped.csv")

        loaded = load_pipeline_clean_df(tier_dir)
        assert loaded.shape == (2, 2)
        assert loaded.columns == ["a", "b"]
```

- [ ] **8.2** Run test, confirm failure:
```bash
pytest tests/test_pipeline_ground_truth.py -v
# Expected: ModuleNotFoundError
```

- [ ] **8.3** Create `dqbench/pipeline_ground_truth.py`:

```python
"""Load and query Pipeline ground truth manifests."""
from __future__ import annotations
import json
from pathlib import Path

import polars as pl
from pydantic import BaseModel


class PipelineGroundTruth(BaseModel):
    tier: int
    version: str
    rows: int                              # rows in messy input
    planted_issues: int                    # quality issues planted
    duplicate_pairs: list[tuple[int, int]] # true match pairs
    expected_output_rows: int              # rows after deduplication


def load_pipeline_ground_truth(path: Path) -> PipelineGroundTruth:
    with open(path) as f:
        return PipelineGroundTruth(**json.load(f))


def load_pipeline_clean_df(tier_dir: Path) -> pl.DataFrame:
    return pl.read_csv(tier_dir / "data_clean_deduped.csv", infer_schema_length=0)
```

- [ ] **8.4** Run ground truth test, confirm pass:
```bash
pytest tests/test_pipeline_ground_truth.py -v
# Expected: 3 passed
```

- [ ] **8.5** Write failing test `tests/test_pipeline_scorer.py`:

```python
"""Tests for Pipeline scorer."""
from __future__ import annotations
import tempfile
from pathlib import Path

import polars as pl
import pytest

from dqbench.pipeline_scorer import score_pipeline_tier
from dqbench.models import PipelineTierResult


class TestScorePipelineTier:
    @pytest.fixture
    def tier_dir(self, tmp_path):
        """Create a tier directory with clean ground truth CSV."""
        clean_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        messy_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4", "5"],
            "name": ["alice", "BOB", "Charlie", "diana", "Eve", "Alice"],
            "email": ["a@x.com", "b@x.com", "C@X.COM", "d@x.com", "e@x.com", "a@x.com"],
        })
        clean_df.write_csv(tmp_path / "data_clean_deduped.csv")
        messy_df.write_csv(tmp_path / "data.csv")
        return tmp_path

    def test_perfect_output(self, tier_dir):
        """Output matches clean ground truth exactly."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        assert isinstance(result, PipelineTierResult)
        assert result.transform_accuracy == 1.0
        assert result.dedup_accuracy == 1.0
        assert result.composite == 1.0

    def test_wrong_row_count(self, tier_dir):
        """Output has too many rows — dedup_accuracy penalized."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4", "5"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Alice"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "a@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        # dedup_accuracy = 1.0 - abs(6 - 5) / 5 = 0.8
        assert result.dedup_accuracy == pytest.approx(0.8)
        assert result.output_rows == 6
        assert result.expected_rows == 5

    def test_wrong_cell_values(self, tier_dir):
        """Some cells not cleaned — transform_accuracy penalized."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["alice", "Bob", "Charlie", "Diana", "Eve"],  # "alice" not fixed
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        assert result.transform_accuracy < 1.0
        assert result.dedup_accuracy == 1.0

    def test_empty_output(self, tier_dir):
        """Empty DataFrame — worst possible scores."""
        result_df = pl.DataFrame({
            "_row_id": pl.Series([], dtype=pl.Utf8),
            "name": pl.Series([], dtype=pl.Utf8),
            "email": pl.Series([], dtype=pl.Utf8),
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        assert result.transform_accuracy == 0.0
        assert result.dedup_accuracy == 0.0
        assert result.composite == 0.0

    def test_composite_calculation(self, tier_dir):
        """Composite = transform_accuracy * 0.6 + dedup_accuracy * 0.4."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        expected_composite = result.transform_accuracy * 0.6 + result.dedup_accuracy * 0.4
        assert result.composite == pytest.approx(expected_composite)
```

- [ ] **8.6** Run test, confirm failure:
```bash
pytest tests/test_pipeline_scorer.py -v
# Expected: ModuleNotFoundError
```

- [ ] **8.7** Create `dqbench/pipeline_scorer.py`:

```python
"""Scoring logic for Pipeline benchmarks."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from dqbench.models import PipelineTierResult
from dqbench.pipeline_ground_truth import load_pipeline_clean_df


def score_pipeline_tier(
    result_df: pl.DataFrame,
    tier_dir: Path,
    tier: int,
    time_seconds: float,
    memory_mb: float,
    expected_rows: int,
) -> PipelineTierResult:
    """Score a pipeline tool's output against the clean+deduplicated ground truth."""
    clean_df = load_pipeline_clean_df(tier_dir)
    messy_df = pl.read_csv(tier_dir / "data.csv", infer_schema_length=0)

    output_rows = result_df.shape[0]

    # ---- Dedup accuracy ----
    if expected_rows == 0:
        dedup_accuracy = 0.0
    else:
        dedup_accuracy = max(0.0, 1.0 - abs(output_rows - expected_rows) / expected_rows)

    # ---- Transform accuracy ----
    # Join result with clean ground truth on _row_id
    if output_rows == 0 or "_row_id" not in result_df.columns:
        transform_accuracy = 0.0
    else:
        # Cast all to string for comparison
        result_str = result_df.cast({col: pl.Utf8 for col in result_df.columns})
        clean_str = clean_df.cast({col: pl.Utf8 for col in clean_df.columns})
        messy_str = messy_df.cast({col: pl.Utf8 for col in messy_df.columns})

        # Join on _row_id
        joined = result_str.join(
            clean_str, on="_row_id", suffix="_clean", how="inner"
        )
        # Also join messy to know which cells were planted
        joined = joined.join(
            messy_str.select([
                pl.col("_row_id"),
                *[pl.col(c).alias(f"{c}_messy") for c in messy_str.columns if c != "_row_id"],
            ]),
            on="_row_id", how="left",
        )

        # Score columns (exclude _row_id)
        score_cols = [c for c in clean_df.columns if c != "_row_id"]
        total_planted = 0
        total_correct = 0

        for col in score_cols:
            if col not in result_str.columns:
                continue
            clean_col = joined[f"{col}_clean"].fill_null("")
            messy_col_name = f"{col}_messy"
            if messy_col_name in joined.columns:
                messy_col = joined[messy_col_name].fill_null("")
            else:
                continue
            result_col = joined[col].fill_null("")

            # Only count cells that differ between messy and clean (planted issues)
            planted_mask = clean_col != messy_col
            planted_count = planted_mask.sum()
            if planted_count == 0:
                continue

            correct_mask = (result_col == clean_col) & planted_mask
            correct = correct_mask.sum()

            total_planted += int(planted_count)
            total_correct += int(correct)

        transform_accuracy = total_correct / total_planted if total_planted > 0 else 0.0

    composite = transform_accuracy * 0.6 + dedup_accuracy * 0.4

    return PipelineTierResult(
        tier=tier,
        transform_accuracy=float(transform_accuracy),
        dedup_accuracy=float(dedup_accuracy),
        composite=float(composite),
        output_rows=output_rows,
        expected_rows=expected_rows,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
    )
```

- [ ] **8.8** Run scorer test, confirm pass:
```bash
pytest tests/test_pipeline_scorer.py -v
# Expected: 5 passed
```

- [ ] **8.9** Create `dqbench/generator/pipeline_tier1.py`:

```python
"""Pipeline Tier 1 dataset generator — quality issues + easy duplicates."""
from __future__ import annotations

import random

import polars as pl

from dqbench.pipeline_ground_truth import PipelineGroundTruth
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COMPANIES,
    STREET_NAMES,
    PHONE_AREA_CODES,
)

NROWS = 1000
N_UNIQUE = 900
N_DUPES = 100
N_ISSUES = 150  # quality issues to plant on non-dupe rows


def _generate_phone(rng: random.Random) -> str:
    area = rng.choice(PHONE_AREA_CODES)
    return f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _generate_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"


def _generate_entity(rng: random.Random) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zipcode = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    email = f"{first.lower()}.{last.lower()}@{domain}"
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": _generate_phone(rng),
        "address": _generate_address(rng),
        "city": city,
        "state": state,
        "zip": zipcode,
        "company": rng.choice(COMPANIES),
    }


def _plant_case_issue(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Lowercase a name field."""
    messy = row.copy()
    field = rng.choice(["first_name", "last_name"])
    messy[field] = messy[field].lower()
    return messy


def _plant_email_issue(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Replace email with an invalid value."""
    messy = row.copy()
    messy["email"] = rng.choice(["N/A", "not provided", "invalid", "---"])
    return messy


def _plant_phone_issue(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    """Corrupt phone format."""
    messy = row.copy()
    messy["phone"] = rng.choice(["555", "N/A", "000-000-0000", ""])
    return messy


def generate_pipeline_tier1() -> tuple[pl.DataFrame, pl.DataFrame, PipelineGroundTruth]:
    """Generate Pipeline Tier 1 dataset.

    Returns:
        (messy_df, clean_deduped_df, ground_truth)
    """
    rng = random.Random(42)

    # Generate unique entities (these are the "clean" records)
    clean_entities: list[dict[str, str]] = []
    for i in range(N_UNIQUE):
        entity = _generate_entity(rng)
        entity["_row_id"] = str(i)
        clean_entities.append(entity)

    # Create messy versions of some entities (quality issues)
    issue_fns = [_plant_case_issue, _plant_email_issue, _plant_phone_issue]
    issue_rows = rng.sample(range(N_UNIQUE), N_ISSUES)
    messy_entities = []
    for i, entity in enumerate(clean_entities):
        if i in issue_rows:
            fn = rng.choice(issue_fns)
            messy_entities.append(fn(entity, rng))
        else:
            messy_entities.append(entity.copy())

    # Create duplicate rows (copies of existing entities with case changes)
    source_indices = rng.sample(range(N_UNIQUE), N_DUPES)
    duplicate_pairs: list[tuple[int, int]] = []

    for src_idx in source_indices:
        dupe_row_idx = len(messy_entities)
        dupe = messy_entities[src_idx].copy()
        dupe["_row_id"] = str(dupe_row_idx)
        dupe["first_name"] = dupe["first_name"].upper()
        dupe["last_name"] = dupe["last_name"].upper()
        messy_entities.append(dupe)
        duplicate_pairs.append((src_idx, dupe_row_idx))

    assert len(messy_entities) == NROWS

    # Clean+deduped = original clean entities only (no dupes)
    clean_deduped_df = pl.DataFrame(clean_entities)

    # Messy = all rows including dupes with issues
    messy_df = pl.DataFrame(messy_entities)

    gt = PipelineGroundTruth(
        tier=1,
        version="1.0.0",
        rows=NROWS,
        planted_issues=N_ISSUES,
        duplicate_pairs=[(min(a, b), max(a, b)) for a, b in duplicate_pairs],
        expected_output_rows=N_UNIQUE,
    )

    return messy_df, clean_deduped_df, gt
```

- [ ] **8.10** Write a quick generator test in `tests/test_pipeline_generator.py`:

```python
"""Tests for Pipeline tier generators."""
from __future__ import annotations
import polars as pl
import pytest

from dqbench.generator.pipeline_tier1 import generate_pipeline_tier1
from dqbench.pipeline_ground_truth import PipelineGroundTruth


class TestPipelineTier1:
    def test_returns_correct_types(self):
        messy, clean, gt = generate_pipeline_tier1()
        assert isinstance(messy, pl.DataFrame)
        assert isinstance(clean, pl.DataFrame)
        assert isinstance(gt, PipelineGroundTruth)

    def test_row_counts(self):
        messy, clean, gt = generate_pipeline_tier1()
        assert messy.shape[0] == 1000
        assert clean.shape[0] == 900
        assert gt.rows == 1000
        assert gt.expected_output_rows == 900

    def test_has_row_id_column(self):
        messy, clean, gt = generate_pipeline_tier1()
        assert "_row_id" in messy.columns
        assert "_row_id" in clean.columns

    def test_duplicate_pair_count(self):
        _, _, gt = generate_pipeline_tier1()
        assert len(gt.duplicate_pairs) == 100

    def test_planted_issues_count(self):
        _, _, gt = generate_pipeline_tier1()
        assert gt.planted_issues == 150

    def test_determinism(self):
        m1, c1, gt1 = generate_pipeline_tier1()
        m2, c2, gt2 = generate_pipeline_tier1()
        assert m1.frame_equal(m2)
        assert c1.frame_equal(c2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, _, gt = generate_pipeline_tier1()
        assert gt.tier == 1
        assert gt.version == "1.0.0"
```

- [ ] **8.11** Run all pipeline tests:
```bash
pytest tests/test_pipeline_ground_truth.py tests/test_pipeline_scorer.py tests/test_pipeline_generator.py -v
# Expected: 15 passed (3 + 5 + 7)
```

- [ ] **8.12** Commit:
```
feat(pipeline): add Pipeline ground truth, Tier 1 generator, and scorer

PipelineGroundTruth Pydantic model with loaders.
Pipeline Tier 1: 1000 rows (900 unique + 100 dupes), 150 planted quality issues.
Pipeline scorer: transform_accuracy (cell-level) * 0.6 + dedup_accuracy * 0.4.
```

---

## Task 9: Pipeline Tier 2, 3 Generators and Runner

**Files:**
- **Create:** `dqbench/generator/pipeline_tier2.py`
- **Create:** `dqbench/generator/pipeline_tier3.py`
- **Modify:** `dqbench/runner.py` (add `ensure_pipeline_datasets`, `run_pipeline_benchmark`)
- **Modify:** `tests/test_pipeline_generator.py` (add Tier 2, 3 tests)
- **Test:** `tests/test_pipeline_runner.py`

### Steps

- [ ] **9.1** Write Tier 2 and 3 generator tests first (see step 9.3 code below), then create `dqbench/generator/pipeline_tier2.py`:

```python
"""Pipeline Tier 2 dataset generator — 5000 rows with quality issues + fuzzy dupes."""
from __future__ import annotations

import random

import polars as pl

from dqbench.pipeline_ground_truth import PipelineGroundTruth
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COMPANIES,
    STREET_NAMES,
    PHONE_AREA_CODES,
    NICKNAME_MAP,
)

NROWS = 5000
N_UNIQUE = 4250
N_DUPES = 750
N_ISSUES = 600


def _generate_phone(rng: random.Random) -> str:
    area = rng.choice(PHONE_AREA_CODES)
    return f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _generate_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"


def _generate_entity(rng: random.Random) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zipcode = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    email = f"{first.lower()}.{last.lower()}@{domain}"
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": _generate_phone(rng),
        "address": _generate_address(rng),
        "city": city,
        "state": state,
        "zip": zipcode,
        "company": rng.choice(COMPANIES),
    }


def _plant_issue(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    messy = row.copy()
    issue_type = rng.choice(["case", "email", "phone", "whitespace", "null"])
    if issue_type == "case":
        field = rng.choice(["first_name", "last_name", "city"])
        messy[field] = messy[field].lower()
    elif issue_type == "email":
        messy["email"] = rng.choice(["N/A", "invalid", "none", "---", ""])
    elif issue_type == "phone":
        messy["phone"] = rng.choice(["N/A", "", "000", "invalid"])
    elif issue_type == "whitespace":
        field = rng.choice(["first_name", "last_name", "company"])
        messy[field] = f"  {messy[field]}  "
    elif issue_type == "null":
        field = rng.choice(["email", "phone", "address"])
        messy[field] = ""
    return messy


def _create_fuzzy_dupe(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    dupe = row.copy()
    strategy = rng.choice(["nickname", "format", "missing", "case"])
    if strategy == "nickname" and dupe["first_name"] in NICKNAME_MAP:
        dupe["first_name"] = rng.choice(NICKNAME_MAP[dupe["first_name"]])
    elif strategy == "format":
        phone = dupe["phone"].replace("(", "").replace(")", "").replace(" ", "").replace("-", "")
        dupe["phone"] = phone
    elif strategy == "missing":
        dupe[rng.choice(["phone", "address", "company"])] = ""
    else:
        dupe["first_name"] = dupe["first_name"].upper()
        dupe["last_name"] = dupe["last_name"].lower()
    return dupe


def generate_pipeline_tier2() -> tuple[pl.DataFrame, pl.DataFrame, PipelineGroundTruth]:
    """Generate Pipeline Tier 2 dataset."""
    rng = random.Random(42)

    clean_entities: list[dict[str, str]] = []
    for i in range(N_UNIQUE):
        entity = _generate_entity(rng)
        entity["_row_id"] = str(i)
        clean_entities.append(entity)

    issue_rows = rng.sample(range(N_UNIQUE), N_ISSUES)
    messy_entities = []
    for i, entity in enumerate(clean_entities):
        if i in issue_rows:
            messy_entities.append(_plant_issue(entity, rng))
        else:
            messy_entities.append(entity.copy())

    source_indices = rng.sample(range(N_UNIQUE), N_DUPES)
    duplicate_pairs: list[tuple[int, int]] = []

    for src_idx in source_indices:
        dupe_row_idx = len(messy_entities)
        dupe = _create_fuzzy_dupe(messy_entities[src_idx], rng)
        dupe["_row_id"] = str(dupe_row_idx)
        messy_entities.append(dupe)
        duplicate_pairs.append((src_idx, dupe_row_idx))

    assert len(messy_entities) == NROWS

    clean_deduped_df = pl.DataFrame(clean_entities)
    messy_df = pl.DataFrame(messy_entities)

    gt = PipelineGroundTruth(
        tier=2,
        version="1.0.0",
        rows=NROWS,
        planted_issues=N_ISSUES,
        duplicate_pairs=[(min(a, b), max(a, b)) for a, b in duplicate_pairs],
        expected_output_rows=N_UNIQUE,
    )

    return messy_df, clean_deduped_df, gt
```

- [ ] **9.2** Create `dqbench/generator/pipeline_tier3.py`:

```python
"""Pipeline Tier 3 dataset generator — 10000 rows with quality issues + adversarial dupes."""
from __future__ import annotations

import random

import polars as pl

from dqbench.pipeline_ground_truth import PipelineGroundTruth
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COMPANIES,
    STREET_NAMES,
    PHONE_AREA_CODES,
    NICKNAME_MAP,
    PHONETIC_VARIANTS,
    ADDRESS_ABBREVIATIONS,
)

NROWS = 10000
N_UNIQUE = 8000
N_DUPES = 2000
N_ISSUES = 1500


def _generate_phone(rng: random.Random) -> str:
    area = rng.choice(PHONE_AREA_CODES)
    return f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _generate_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREET_NAMES)}"


def _generate_entity(rng: random.Random) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    city, state, zipcode = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    email = f"{first.lower()}.{last.lower()}@{domain}"
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": _generate_phone(rng),
        "address": _generate_address(rng),
        "city": city,
        "state": state,
        "zip": zipcode,
        "company": rng.choice(COMPANIES),
    }


def _plant_issue(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    messy = row.copy()
    issue_type = rng.choice([
        "case", "email", "phone", "whitespace", "null",
        "unicode", "swap", "abbreviation",
    ])
    if issue_type == "case":
        field = rng.choice(["first_name", "last_name", "city", "company"])
        messy[field] = messy[field].lower() if rng.random() < 0.5 else messy[field].upper()
    elif issue_type == "email":
        messy["email"] = rng.choice(["N/A", "invalid", "none", "", "---", "not@valid"])
    elif issue_type == "phone":
        messy["phone"] = rng.choice(["N/A", "", "000", "invalid", "555"])
    elif issue_type == "whitespace":
        field = rng.choice(["first_name", "last_name", "company", "address"])
        messy[field] = f"  {messy[field]}  "
    elif issue_type == "null":
        field = rng.choice(["email", "phone", "address", "company"])
        messy[field] = ""
    elif issue_type == "unicode":
        field = rng.choice(["first_name", "last_name"])
        val = messy[field]
        if len(val) > 2:
            idx = rng.randint(1, len(val) - 1)
            messy[field] = val[:idx] + "\u200b" + val[idx:]
    elif issue_type == "swap":
        messy["first_name"], messy["last_name"] = messy["last_name"], messy["first_name"]
    elif issue_type == "abbreviation":
        addr = messy["address"]
        for abbr, full in ADDRESS_ABBREVIATIONS.items():
            if addr.endswith(abbr):
                messy["address"] = addr[: -len(abbr)] + full
                break
    return messy


def _create_adversarial_dupe(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    dupe = row.copy()
    strategy = rng.choice([
        "phonetic", "unicode", "split", "abbreviation", "multi_field",
    ])
    if strategy == "phonetic":
        last = dupe["last_name"]
        if last in PHONETIC_VARIANTS:
            dupe["last_name"] = rng.choice(PHONETIC_VARIANTS[last])
        else:
            name = list(dupe["last_name"])
            if len(name) > 2:
                idx = rng.randint(1, len(name) - 1)
                name.insert(idx, "h")
                dupe["last_name"] = "".join(name)
    elif strategy == "unicode":
        name = dupe["first_name"]
        if len(name) > 2:
            idx = rng.randint(1, len(name) - 1)
            dupe["first_name"] = name[:idx] + "\u200b" + name[idx:]
    elif strategy == "split":
        for field in ["phone", "address", "city", "state", "zip"]:
            if rng.random() < 0.5:
                dupe[field] = ""
    elif strategy == "abbreviation":
        addr = dupe["address"]
        for abbr, full in ADDRESS_ABBREVIATIONS.items():
            if addr.endswith(abbr):
                dupe["address"] = addr[: -len(abbr)] + full
                break
    elif strategy == "multi_field":
        dupe["first_name"] = dupe["first_name"].lower()
        dupe["last_name"] = dupe["last_name"].upper()
        phone = dupe["phone"].replace("(", "").replace(")", "").replace(" ", "").replace("-", "")
        dupe["phone"] = phone
    return dupe


def generate_pipeline_tier3() -> tuple[pl.DataFrame, pl.DataFrame, PipelineGroundTruth]:
    """Generate Pipeline Tier 3 dataset."""
    rng = random.Random(42)

    clean_entities: list[dict[str, str]] = []
    for i in range(N_UNIQUE):
        entity = _generate_entity(rng)
        entity["_row_id"] = str(i)
        clean_entities.append(entity)

    issue_rows = set(rng.sample(range(N_UNIQUE), N_ISSUES))
    messy_entities = []
    for i, entity in enumerate(clean_entities):
        if i in issue_rows:
            messy_entities.append(_plant_issue(entity, rng))
        else:
            messy_entities.append(entity.copy())

    source_indices = rng.sample(range(N_UNIQUE), N_DUPES)
    duplicate_pairs: list[tuple[int, int]] = []

    for src_idx in source_indices:
        dupe_row_idx = len(messy_entities)
        dupe = _create_adversarial_dupe(messy_entities[src_idx], rng)
        dupe["_row_id"] = str(dupe_row_idx)
        messy_entities.append(dupe)
        duplicate_pairs.append((src_idx, dupe_row_idx))

    assert len(messy_entities) == NROWS

    clean_deduped_df = pl.DataFrame(clean_entities)
    messy_df = pl.DataFrame(messy_entities)

    gt = PipelineGroundTruth(
        tier=3,
        version="1.0.0",
        rows=NROWS,
        planted_issues=N_ISSUES,
        duplicate_pairs=[(min(a, b), max(a, b)) for a, b in duplicate_pairs],
        expected_output_rows=N_UNIQUE,
    )

    return messy_df, clean_deduped_df, gt
```

- [ ] **9.3** Add Tier 2 and 3 test classes to `tests/test_pipeline_generator.py`:

```python
from dqbench.generator.pipeline_tier2 import generate_pipeline_tier2
from dqbench.generator.pipeline_tier3 import generate_pipeline_tier3


class TestPipelineTier2:
    def test_returns_correct_types(self):
        messy, clean, gt = generate_pipeline_tier2()
        assert isinstance(messy, pl.DataFrame)
        assert isinstance(clean, pl.DataFrame)
        assert isinstance(gt, PipelineGroundTruth)

    def test_row_counts(self):
        messy, clean, gt = generate_pipeline_tier2()
        assert messy.shape[0] == 5000
        assert clean.shape[0] == 4250
        assert gt.rows == 5000
        assert gt.expected_output_rows == 4250

    def test_has_row_id_column(self):
        messy, clean, _ = generate_pipeline_tier2()
        assert "_row_id" in messy.columns
        assert "_row_id" in clean.columns

    def test_duplicate_pair_count(self):
        _, _, gt = generate_pipeline_tier2()
        assert len(gt.duplicate_pairs) == 750

    def test_determinism(self):
        m1, c1, gt1 = generate_pipeline_tier2()
        m2, c2, gt2 = generate_pipeline_tier2()
        assert m1.frame_equal(m2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, _, gt = generate_pipeline_tier2()
        assert gt.tier == 2


class TestPipelineTier3:
    def test_returns_correct_types(self):
        messy, clean, gt = generate_pipeline_tier3()
        assert isinstance(messy, pl.DataFrame)
        assert isinstance(clean, pl.DataFrame)
        assert isinstance(gt, PipelineGroundTruth)

    def test_row_counts(self):
        messy, clean, gt = generate_pipeline_tier3()
        assert messy.shape[0] == 10000
        assert clean.shape[0] == 8000
        assert gt.rows == 10000
        assert gt.expected_output_rows == 8000

    def test_has_row_id_column(self):
        messy, clean, _ = generate_pipeline_tier3()
        assert "_row_id" in messy.columns
        assert "_row_id" in clean.columns

    def test_duplicate_pair_count(self):
        _, _, gt = generate_pipeline_tier3()
        assert len(gt.duplicate_pairs) == 2000

    def test_determinism(self):
        m1, c1, gt1 = generate_pipeline_tier3()
        m2, c2, gt2 = generate_pipeline_tier3()
        assert m1.frame_equal(m2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, _, gt = generate_pipeline_tier3()
        assert gt.tier == 3
```

- [ ] **9.4** Add `ensure_pipeline_datasets()` and `run_pipeline_benchmark()` to `dqbench/runner.py`. Add imports:

```python
from dqbench.adapters.base import PipelineAdapter
from dqbench.models import PipelineScorecard
```

Then add:

```python
def ensure_pipeline_datasets() -> None:
    """Generate Pipeline datasets if not cached."""
    if (CACHE_DIR / "pipeline_tier1" / "data.csv").exists():
        return
    from dqbench.generator.pipeline_tier1 import generate_pipeline_tier1

    generators = [(1, generate_pipeline_tier1)]

    try:
        from dqbench.generator.pipeline_tier2 import generate_pipeline_tier2
        generators.append((2, generate_pipeline_tier2))
    except ImportError:
        pass
    try:
        from dqbench.generator.pipeline_tier3 import generate_pipeline_tier3
        generators.append((3, generate_pipeline_tier3))
    except ImportError:
        pass

    for tier_num, gen_fn in generators:
        tier_dir = CACHE_DIR / f"pipeline_tier{tier_num}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        messy_df, clean_df, gt = gen_fn()
        messy_df.write_csv(tier_dir / "data.csv")
        clean_df.write_csv(tier_dir / "data_clean_deduped.csv")
        with open(tier_dir / "pipeline_ground_truth.json", "w") as f:
            json.dump(gt.model_dump(), f, indent=2)


def run_pipeline_benchmark(
    adapter: PipelineAdapter,
    tiers: list[int] | None = None,
) -> PipelineScorecard:
    """Run Pipeline benchmark and return a scorecard."""
    from dqbench.pipeline_scorer import score_pipeline_tier
    from dqbench.pipeline_ground_truth import load_pipeline_ground_truth

    ensure_pipeline_datasets()
    tier_nums = tiers or [1, 2, 3]
    results = []

    for tier_num in tier_nums:
        tier_dir = CACHE_DIR / f"pipeline_tier{tier_num}"
        csv_path = tier_dir / "data.csv"
        gt_path = tier_dir / "pipeline_ground_truth.json"

        if not csv_path.exists():
            continue

        gt = load_pipeline_ground_truth(gt_path)

        tracemalloc.start()
        t0 = time.perf_counter()
        result_df = adapter.run_pipeline(csv_path)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tier_result = score_pipeline_tier(
            result_df, tier_dir, tier=tier_num,
            time_seconds=elapsed, memory_mb=peak / (1024 * 1024),
            expected_rows=gt.expected_output_rows,
        )
        results.append(tier_result)

    return PipelineScorecard(
        tool_name=adapter.name,
        tool_version=adapter.version,
        tiers=results,
    )
```

- [ ] **9.5** Write test `tests/test_pipeline_runner.py`:

```python
"""Tests for Pipeline runner integration."""
from __future__ import annotations
import json
from pathlib import Path

import polars as pl
import pytest

from dqbench.adapters.base import PipelineAdapter
from dqbench.models import PipelineScorecard


class PassthroughPipelineAdapter(PipelineAdapter):
    """Returns the clean+deduped ground truth as the pipeline output."""

    @property
    def name(self) -> str:
        return "passthrough-pipeline"

    @property
    def version(self) -> str:
        return "0.0.1"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        # Load the clean ground truth (cheating for test purposes)
        clean_path = csv_path.parent / "data_clean_deduped.csv"
        return pl.read_csv(clean_path, infer_schema_length=0)


class TestPipelineRunner:
    def test_run_pipeline_benchmark_returns_scorecard(self):
        from dqbench.runner import run_pipeline_benchmark, ensure_pipeline_datasets
        ensure_pipeline_datasets()
        adapter = PassthroughPipelineAdapter()
        scorecard = run_pipeline_benchmark(adapter, tiers=[1])
        assert isinstance(scorecard, PipelineScorecard)
        assert scorecard.tool_name == "passthrough-pipeline"
        assert len(scorecard.tiers) == 1

    def test_perfect_pipeline_scores(self):
        from dqbench.runner import run_pipeline_benchmark, ensure_pipeline_datasets
        ensure_pipeline_datasets()
        adapter = PassthroughPipelineAdapter()
        scorecard = run_pipeline_benchmark(adapter, tiers=[1])
        assert scorecard.tiers[0].transform_accuracy == 1.0
        assert scorecard.tiers[0].dedup_accuracy == 1.0
        assert scorecard.tiers[0].composite == 1.0

    def test_ensure_pipeline_datasets_creates_files(self):
        from dqbench.runner import CACHE_DIR, ensure_pipeline_datasets
        ensure_pipeline_datasets()
        assert (CACHE_DIR / "pipeline_tier1" / "data.csv").exists()
        assert (CACHE_DIR / "pipeline_tier1" / "data_clean_deduped.csv").exists()
        assert (CACHE_DIR / "pipeline_tier1" / "pipeline_ground_truth.json").exists()
```

- [ ] **9.6** Run all pipeline tests:
```bash
pytest tests/test_pipeline_generator.py tests/test_pipeline_runner.py -v
# Expected: 19 passed (7 tier1 + 6 tier2 + 6 tier3 from generator, 3 from runner)
```

- [ ] **9.7** Commit:
```
feat(pipeline): add Pipeline Tier 2/3 generators and runner

Tier 2: 5000 rows, 750 dupes, 600 quality issues.
Tier 3: 10000 rows, 2000 dupes, 1500 quality issues.
Runner follows run_transform_benchmark pattern with ensure_pipeline_datasets().
```

---

## Task 10: CLI Integration, Adapters, and Report

**Files:**
- **Modify:** `dqbench/cli.py`
- **Create:** `dqbench/adapters/goldenmatch_adapter.py`
- **Create:** `dqbench/adapters/goldenpipe_adapter.py`
- **Modify:** `dqbench/report.py`
- **Test:** `tests/test_cli_er_pipeline.py`

### Steps

- [ ] **10.1** Create `dqbench/adapters/goldenmatch_adapter.py`:

```python
"""GoldenMatch adapter for DQBench ER benchmarks."""
from __future__ import annotations
from pathlib import Path

from dqbench.adapters.base import EntityResolutionAdapter


class GoldenMatchAdapter(EntityResolutionAdapter):
    @property
    def name(self) -> str:
        return "GoldenMatch"

    @property
    def version(self) -> str:
        try:
            import goldenmatch
            return goldenmatch.__version__
        except (ImportError, AttributeError):
            return "0.0.0"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        try:
            from goldenmatch import deduplicate
            return deduplicate(csv_path)
        except ImportError:
            raise ImportError(
                "goldenmatch is not installed. "
                "Install it with: pip install goldenmatch"
            )
```

- [ ] **10.2** Create `dqbench/adapters/goldenpipe_adapter.py`:

```python
"""GoldenPipe adapter for DQBench Pipeline benchmarks."""
from __future__ import annotations
from pathlib import Path

import polars as pl

from dqbench.adapters.base import PipelineAdapter


class GoldenPipeAdapter(PipelineAdapter):
    @property
    def name(self) -> str:
        return "GoldenPipe"

    @property
    def version(self) -> str:
        try:
            import goldenpipe
            return goldenpipe.__version__
        except (ImportError, AttributeError):
            return "0.0.0"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        try:
            from goldenpipe import run_pipeline
            return run_pipeline(csv_path)
        except ImportError:
            raise ImportError(
                "goldenpipe is not installed. "
                "Install it with: pip install goldenpipe"
            )
```

- [ ] **10.3** Modify `dqbench/cli.py` — update BUILTIN_ADAPTERS:

Add to the `BUILTIN_ADAPTERS` dict:
```python
    # GoldenMatch (ER)
    "goldenmatch": "dqbench.adapters.goldenmatch_adapter:GoldenMatchAdapter",
    # GoldenPipe (Pipeline)
    "goldenpipe": "dqbench.adapters.goldenpipe_adapter:GoldenPipeAdapter",
```

Add adapter category lists after ALL_ADAPTER_NAMES:
```python
DETECT_ADAPTERS = [
    "goldencheck", "gx-zero", "gx-auto", "gx-best",
    "pandera-zero", "pandera-auto", "pandera-best",
    "soda-zero", "soda-auto", "soda-best",
]
TRANSFORM_ADAPTERS = ["goldenflow"]
ER_ADAPTERS = ["goldenmatch"]
PIPELINE_ADAPTERS = ["goldenpipe"]
```

- [ ] **10.4** Add `_detect_category()` function to `dqbench/cli.py`:

```python
def _detect_category(adapter) -> str:
    """Detect benchmark category from adapter type."""
    from dqbench.adapters.base import (
        PipelineAdapter,
        EntityResolutionAdapter,
        TransformAdapter,
    )
    if isinstance(adapter, PipelineAdapter):
        return "pipeline"
    if isinstance(adapter, EntityResolutionAdapter):
        return "er"
    if isinstance(adapter, TransformAdapter):
        return "transform"
    return "detect"
```

- [ ] **10.5** Update the `run` command to use `_detect_category()` for dispatch:

Replace the `isinstance(adapter, TransformAdapter)` block in the `run` command with:

```python
    category = _detect_category(adapter)
    if category == "er":
        from dqbench.runner import run_er_benchmark
        from dqbench.report import report_er_rich, report_er_json
        scorecard = run_er_benchmark(adapter, tiers=tiers)
        if json_output:
            report_er_json(scorecard, sys.stdout)
        else:
            report_er_rich(scorecard)
    elif category == "pipeline":
        from dqbench.runner import run_pipeline_benchmark
        from dqbench.report import report_pipeline_rich, report_pipeline_json
        scorecard = run_pipeline_benchmark(adapter, tiers=tiers)
        if json_output:
            report_pipeline_json(scorecard, sys.stdout)
        else:
            report_pipeline_rich(scorecard)
    elif category == "transform":
        from dqbench.runner import run_transform_benchmark
        from dqbench.report import report_transform_rich, report_transform_json
        scorecard = run_transform_benchmark(adapter, tiers=tiers)
        if json_output:
            report_transform_json(scorecard, sys.stdout)
        else:
            report_transform_rich(scorecard)
    else:
        scorecard = run_benchmark(adapter, tiers=tiers)
        if json_output:
            from dqbench.report import report_json
            report_json(scorecard, sys.stdout)
        else:
            from dqbench.report import report_rich
            report_rich(scorecard)
```

- [ ] **10.6** Update the `run` command signature to accept `--real` flag:

```python
@app.command()
def run(
    adapter_name: str = typer.Argument(..., help=(
        "Adapter name: goldencheck | goldenmatch | goldenpipe | ... | all"
    )),
    tier: Optional[int] = typer.Option(None, "--tier", "-t", help="Run specific tier only (1, 2, or 3)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    adapter_path: Optional[Path] = typer.Option(None, "--adapter", help="Path to custom adapter file"),
    real: bool = typer.Option(False, "--real", help="Include real datasets (ER only)"),
    er: bool = typer.Option(False, "--er", help="Run all ER adapters (with 'all')"),
    pipeline: bool = typer.Option(False, "--pipeline", help="Run all Pipeline adapters (with 'all')"),
) -> None:
```

- [ ] **10.7** Update the `_run_all` function to support category filtering:

```python
def _run_all(tier: Optional[int] = None, category: str = "detect") -> None:
    """Run all registered adapters in a category and print a comparison table."""
    adapters_map = {
        "detect": DETECT_ADAPTERS,
        "transform": TRANSFORM_ADAPTERS,
        "er": ER_ADAPTERS,
        "pipeline": PIPELINE_ADAPTERS,
    }
    adapter_names = adapters_map.get(category, DETECT_ADAPTERS)
    tiers = [tier] if tier else None

    scorecards = []
    for name in adapter_names:
        typer.echo(f"\nRunning: {name} ...", err=True)
        try:
            adapter = _load_adapter(name)
            cat = _detect_category(adapter)
            if cat == "er":
                from dqbench.runner import run_er_benchmark
                sc = run_er_benchmark(adapter, tiers=tiers)
                typer.echo(f"  Done — score: {sc.dqbench_er_score:.2f}", err=True)
            elif cat == "pipeline":
                from dqbench.runner import run_pipeline_benchmark
                sc = run_pipeline_benchmark(adapter, tiers=tiers)
                typer.echo(f"  Done — score: {sc.dqbench_pipeline_score:.2f}", err=True)
            elif cat == "transform":
                from dqbench.runner import run_transform_benchmark
                sc = run_transform_benchmark(adapter, tiers=tiers)
                typer.echo(f"  Done — score: {sc.composite_score:.2f}", err=True)
            else:
                from dqbench.runner import run_benchmark
                sc = run_benchmark(adapter, tiers=tiers)
                typer.echo(f"  Done — score: {sc.dqbench_score:.2f}", err=True)
            scorecards.append(sc)
        except Exception as e:
            typer.echo(f"  FAILED: {e}", err=True)

    if scorecards:
        # Use category-specific comparison reports (scorecards are all same type within a category)
        if category == "er":
            from dqbench.report import report_er_comparison
            report_er_comparison(scorecards)
        elif category == "pipeline":
            from dqbench.report import report_pipeline_comparison
            report_pipeline_comparison(scorecards)
        elif category == "transform":
            from dqbench.report import report_transform_comparison
            report_transform_comparison(scorecards)
        else:
            from dqbench.report import report_comparison
            report_comparison(scorecards)
```

And update the `all` dispatch in `run`:
```python
    if adapter_name == "all":
        if er:
            _run_all(tier=tier, category="er")
        elif pipeline:
            _run_all(tier=tier, category="pipeline")
        else:
            _run_all(tier=tier)
        return
```

- [ ] **10.8** Update the `generate` command to support `--er`, `--pipeline`, `--real`, `--all` flags:

```python
@app.command()
def generate(
    force: bool = typer.Option(False, "--force", help="Regenerate even if cached"),
    er: bool = typer.Option(False, "--er", help="Generate ER datasets"),
    pipeline: bool = typer.Option(False, "--pipeline", help="Generate Pipeline datasets"),
    real: bool = typer.Option(False, "--real", help="Download real ER datasets"),
    all_datasets: bool = typer.Option(False, "--all", help="Generate all datasets"),
) -> None:
    """Generate benchmark datasets."""
    from dqbench.runner import CACHE_DIR, ensure_datasets

    if force:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

    gen_detect = all_datasets or (not er and not pipeline and not real)

    if gen_detect or all_datasets:
        ensure_datasets()
        typer.echo(f"Detect datasets generated at {CACHE_DIR}")

    if er or all_datasets:
        from dqbench.runner import ensure_er_datasets
        ensure_er_datasets()
        typer.echo(f"ER datasets generated at {CACHE_DIR}")

    if pipeline or all_datasets:
        from dqbench.runner import ensure_pipeline_datasets
        ensure_pipeline_datasets()
        typer.echo(f"Pipeline datasets generated at {CACHE_DIR}")

    if real or all_datasets:
        typer.echo("Real dataset download not yet implemented.")
```

- [ ] **10.9** Update `_load_adapter` to detect all adapter types:

In the custom adapter path block, update the isinstance checks:
```python
    if path:
        import importlib.util
        from dqbench.adapters.base import BenchmarkAdapter

        spec = importlib.util.spec_from_file_location("custom_adapter", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, BenchmarkAdapter) and obj is not BenchmarkAdapter:
                return obj()
        raise typer.Exit("No BenchmarkAdapter subclass found in adapter file.")
```

- [ ] **10.10** Add ER and Pipeline report functions to `dqbench/report.py`:

```python
from dqbench.models import Scorecard, TransformScorecard, ERScorecard, PipelineScorecard


def report_er_rich(scorecard: ERScorecard) -> None:
    """Print ER scorecard using Rich."""
    console = Console()

    console.print()
    console.print("[bold cyan]DQBench ER Report[/bold cyan]", justify="center")
    console.print(f"[bold]Tool:[/bold] {scorecard.tool_name}  [bold]Version:[/bold] {scorecard.tool_version}")
    console.print()

    table = Table(
        title="Entity Resolution — Pair-Level Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Tier", style="bold", justify="center")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for t in scorecard.tiers:
        table.add_row(
            str(t.tier),
            f"{t.precision:.1%}",
            f"{t.recall:.1%}",
            f"{t.f1:.1%}",
            str(t.false_positives),
            str(t.false_negatives),
            f"{t.time_seconds:.3f}",
            f"{t.memory_mb:.1f}",
        )

    console.print(table)
    console.print()
    console.print(f"[bold green]DQBench ER Score: {scorecard.dqbench_er_score:.2f}[/bold green]", justify="center")
    console.print()

    if scorecard.real_datasets:
        real_table = Table(
            title="Real Datasets (not in composite score)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
        )
        real_table.add_column("Dataset", style="bold")
        real_table.add_column("Precision", justify="right")
        real_table.add_column("Recall", justify="right")
        real_table.add_column("F1", justify="right")
        real_table.add_column("Time (s)", justify="right")

        for r in scorecard.real_datasets:
            real_table.add_row(
                r.dataset_name,
                f"{r.precision:.1%}",
                f"{r.recall:.1%}",
                f"{r.f1:.1%}",
                f"{r.time_seconds:.3f}",
            )
        console.print(real_table)
        console.print()


def report_er_json(scorecard: ERScorecard, stream: IO) -> None:
    """Write ER scorecard as JSON."""
    data = {
        "tool_name": scorecard.tool_name,
        "tool_version": scorecard.tool_version,
        "dqbench_er_score": scorecard.dqbench_er_score,
        "tiers": [dataclasses.asdict(t) for t in scorecard.tiers],
        "real_datasets": [dataclasses.asdict(r) for r in scorecard.real_datasets] if scorecard.real_datasets else None,
    }
    json.dump(data, stream, indent=2)
    stream.write("\n")


def report_pipeline_rich(scorecard: PipelineScorecard) -> None:
    """Print Pipeline scorecard using Rich."""
    console = Console()

    console.print()
    console.print("[bold cyan]DQBench Pipeline Report[/bold cyan]", justify="center")
    console.print(f"[bold]Tool:[/bold] {scorecard.tool_name}  [bold]Version:[/bold] {scorecard.tool_version}")
    console.print()

    table = Table(
        title="Pipeline — Transform + Dedup Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Tier", style="bold", justify="center")
    table.add_column("Transform Acc", justify="right")
    table.add_column("Dedup Acc", justify="right")
    table.add_column("Composite", justify="right")
    table.add_column("Output Rows", justify="right")
    table.add_column("Expected Rows", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for t in scorecard.tiers:
        table.add_row(
            str(t.tier),
            f"{t.transform_accuracy:.1%}",
            f"{t.dedup_accuracy:.1%}",
            f"{t.composite:.1%}",
            str(t.output_rows),
            str(t.expected_rows),
            f"{t.time_seconds:.3f}",
            f"{t.memory_mb:.1f}",
        )

    console.print(table)
    console.print()
    console.print(
        f"[bold green]DQBench Pipeline Score: {scorecard.dqbench_pipeline_score:.2f}[/bold green]",
        justify="center",
    )
    console.print()


def report_pipeline_json(scorecard: PipelineScorecard, stream: IO) -> None:
    """Write Pipeline scorecard as JSON."""
    data = {
        "tool_name": scorecard.tool_name,
        "tool_version": scorecard.tool_version,
        "dqbench_pipeline_score": scorecard.dqbench_pipeline_score,
        "tiers": [dataclasses.asdict(t) for t in scorecard.tiers],
    }
    json.dump(data, stream, indent=2)
    stream.write("\n")
```

- [ ] **10.11** Write test `tests/test_cli_er_pipeline.py`:

```python
"""Tests for CLI ER and Pipeline integration."""
from __future__ import annotations
import pytest

from dqbench.cli import _detect_category, BUILTIN_ADAPTERS


def test_goldenmatch_in_builtin_adapters():
    assert "goldenmatch" in BUILTIN_ADAPTERS


def test_goldenpipe_in_builtin_adapters():
    assert "goldenpipe" in BUILTIN_ADAPTERS


def test_detect_category_er():
    from dqbench.adapters.base import EntityResolutionAdapter
    from pathlib import Path

    class FakeER(EntityResolutionAdapter):
        @property
        def name(self): return "x"
        @property
        def version(self): return "0"
        def deduplicate(self, csv_path: Path): return []

    assert _detect_category(FakeER()) == "er"


def test_detect_category_pipeline():
    from dqbench.adapters.base import PipelineAdapter
    from pathlib import Path
    import polars as pl

    class FakePipeline(PipelineAdapter):
        @property
        def name(self): return "x"
        @property
        def version(self): return "0"
        def run_pipeline(self, csv_path: Path): return pl.DataFrame()

    assert _detect_category(FakePipeline()) == "pipeline"


def test_detect_category_transform():
    from dqbench.adapters.base import TransformAdapter
    from pathlib import Path
    import polars as pl

    class FakeTransform(TransformAdapter):
        @property
        def name(self): return "x"
        @property
        def version(self): return "0"
        def transform(self, csv_path: Path): return pl.DataFrame()

    assert _detect_category(FakeTransform()) == "transform"


def test_detect_category_detect():
    from dqbench.adapters.base import DQBenchAdapter
    from pathlib import Path

    class FakeDetect(DQBenchAdapter):
        @property
        def name(self): return "x"
        @property
        def version(self): return "0"
        def validate(self, csv_path: Path): return []

    assert _detect_category(FakeDetect()) == "detect"
```

- [ ] **10.12** Run CLI tests:
```bash
pytest tests/test_cli_er_pipeline.py -v
# Expected: 6 passed
```

- [ ] **10.13** Run full test suite to verify nothing is broken:
```bash
pytest tests/ -v
# Expected: All tests pass, including existing detect/transform tests
```

- [ ] **10.14** Commit:
```
feat(cli): integrate ER and Pipeline benchmarks into CLI

Add goldenmatch and goldenpipe to BUILTIN_ADAPTERS.
Add _detect_category() for automatic adapter dispatch.
Add --er, --pipeline, --real, --all flags to generate and run commands.
Add ER and Pipeline Rich/JSON report functions.
```

---

## Dependency Graph

```
Task 1 (adapters/base.py)
   |
   v
Task 2 (models.py) -----> Task 3 (er_ground_truth.py)
   |                          |
   |                          v
   |                       Task 4 (er_tier1 generator)
   |                          |
   |                          v
   |                       Task 5 (er_scorer.py)
   |                          |
   |                          v
   |                       Task 6 (er runner)
   |                          |
   |                          v
   |                       Task 7 (er_tier2, er_tier3)
   |
   +-----> Task 8 (pipeline ground truth, tier1, scorer)
              |
              v
           Task 9 (pipeline tier2, tier3, runner)
              |
              v
           Task 10 (CLI, adapters, report) <--- depends on all above
```

Tasks 3-7 (ER track) and Task 8-9 (Pipeline track) can be parallelized after Tasks 1-2 are complete, since they have no cross-dependencies until Task 10.

---

## Verification Checklist

After all tasks are complete:

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `dqbench generate --all` — creates detect, ER, and pipeline datasets
- [ ] `dqbench generate --er` — creates only ER datasets
- [ ] `dqbench generate --pipeline` — creates only pipeline datasets
- [ ] `python -c "from dqbench.adapters.base import BenchmarkAdapter, DQBenchAdapter, TransformAdapter, EntityResolutionAdapter, PipelineAdapter; print('OK')"` — all imports work
- [ ] Existing detect/transform tests still pass unchanged
- [ ] All generators are deterministic (run twice, compare output)
