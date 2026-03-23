# DQBench Design Spec

**Date:** 2026-03-23
**Author:** Ben Severn
**Status:** Draft

## Overview

DQBench is a standalone, tool-agnostic data quality benchmark package. It ships deterministic datasets with planted quality issues and ground truth, a simple adapter interface for any validation tool, and a scoring framework that measures recall, precision, and false positive rate across three difficulty tiers.

**One-liner:** "The ImageNet of data quality — standardized benchmarks for validation tools."

## Target Users

1. **Validation tool developers** — benchmark their tool against a standard
2. **Data engineers evaluating tools** — compare GoldenCheck vs. Great Expectations vs. Pandera on the same datasets
3. **Researchers** — cite DQBench in papers about data quality

## Package Name

`dqbench` — `pip install dqbench`

## Architecture

```
dqbench/
├── cli.py              # Typer CLI: run, generate, results
├── runner.py           # Orchestrates adapter against all tiers
├── scorer.py           # Computes recall, precision, F1, FPR, DQBench Score
├── report.py           # Rich console + JSON scorecard output
├── ground_truth.py     # Load/query ground truth manifests
├── generator/
│   ├── __init__.py
│   ├── tier1.py        # Basics dataset generator
│   ├── tier2.py        # Realistic dataset generator
│   ├── tier3.py        # Adversarial dataset generator
│   └── utils.py        # Shared generation utilities (fake names, emails, etc.)
├── adapters/
│   ├── base.py         # Abstract adapter interface
│   └── goldencheck.py  # Built-in GoldenCheck adapter
├── models.py           # DQBenchFinding, TierResult, Scorecard dataclasses
└── datasets/           # Generated at runtime, cached locally
    ├── tier1/
    │   ├── data.csv
    │   └── ground_truth.json
    ├── tier2/
    │   ├── data.csv
    │   └── ground_truth.json
    └── tier3/
        ├── data.csv
        └── ground_truth.json
```

## Adapter Interface

```python
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DQBenchFinding:
    """A single finding from a validation tool."""
    column: str
    severity: str       # "error", "warning", "info"
    check: str          # what kind of issue (e.g., "nullability", "format")
    message: str        # human-readable description
    confidence: float = 1.0  # 0.0-1.0, optional


class DQBenchAdapter(ABC):
    """Implement this to benchmark your validation tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for display."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Tool version for display."""
        ...

    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        """Run your tool on this CSV file. Return all findings."""
        ...
```

20 lines to integrate any tool.

**Scoring note:** Column-level matching only — `DQBenchFinding.check` is informational/display-only in v1.0 scoring. A column is "detected" if any finding targets it, regardless of check type.

## CLI Interface

```bash
# Run benchmark with built-in adapter
dqbench run goldencheck

# Run with custom adapter file
dqbench run --adapter path/to/my_adapter.py

# Run specific tier
dqbench run goldencheck --tier 2

# JSON output
dqbench run goldencheck --json

# Generate/regenerate datasets
dqbench generate

# Show last results
dqbench results

# List available adapters
dqbench list
```

## Tier 1: Basics (5K rows, 15 columns)

**Purpose:** Baseline — if your tool can't hit 80% F1 here, it's broken.

**Domain:** Customer database.

**Columns:**
| Column | Type | Planted Issues |
|--------|------|---------------|
| customer_id | int | 15 duplicates |
| first_name | string | 8 nulls in required column |
| last_name | string | 5 numeric values ("12345") |
| email | email | 25 non-email values |
| phone | phone | 3 format variants mixed |
| age | int (stored as string) | 20 string values ("thirty"), 8 outliers (999, -1) |
| income | float | 10 extreme outliers (9999999) |
| status | enum | 15 misspelled variants ("actve", "ACTIVE") |
| signup_date | date | 12 wrong format (MM-DD-YYYY vs YYYY-MM-DD) |
| last_login | date | 18 violations where last_login < signup_date |
| country | 2-letter code | 10 invalid codes ("XX", "ZZ") |
| zip_code | string | mixed 5-digit and 9-digit formats |
| shipping_address | string | 50 correlated nulls with city/zip |
| shipping_city | string | correlated nulls |
| shipping_zip | string | correlated nulls |

**Clean columns (false positive traps — 5 columns):**
| Column | Type | Why It's a Trap |
|--------|------|-----------------|
| order_count | int | Legitimate low values (0-3), looks like it could be missing data |
| account_type | string | Low cardinality (3 values) but intentionally valid |
| last_updated | datetime | Always >= signup_date, valid timestamps with timezone |
| notes | string | 70% null (legitimately optional), free text when present |
| referral_source | string | Mixed casing ("Google", "google", "GOOGLE") — valid, not an error |

**Total planted issues:** ~200
**Total columns:** 20 (15 with issues, 5 clean)

## Tier 2: Realistic (50K rows, 30 columns)

**Purpose:** Tests both detection AND false positive control. Legitimately messy columns that should NOT be flagged.

**Domain:** E-commerce orders with customer info.

**Columns (30):**
- 15 columns WITH planted issues (specific examples — full table defined in generator code):

| Column | Planted Issue | Count | Difficulty |
|--------|--------------|-------|------------|
| order_total | Outliers at 3.1 stddev (near threshold) | 20 | Near-threshold |
| customer_email | 0.4% invalid format (low frequency) | 200 | Low-frequency |
| product_category | Drift: first 40K rows have 10 categories, last 10K introduce 3 new ones | 10K rows | Gradual drift |
| website_url | 15 values in email format instead of URL | 15 | Wrong context |
| billing_zip | Stored as Int64 instead of String (loses leading zeros) | all rows | Type ambiguity |
| phone_number | 0.2% contain letters mixed in | 100 | Near-invisible |
| ship_date | 30 rows where ship_date < order_date | 30 | Temporal |
| quantity | 5 negative values | 5 | Semantic |
| discount_pct | 8 values > 100% | 8 | Range |
| customer_name | 12 rows with numeric strings | 12 | Minority type |
| sku | 25 duplicates in unique column | 25 | Near-unique |
| rating | Values 1-5 plus 8 values of "6" (enum violation) | 8 | Enum drift |
| address_line1 | 40 correlated nulls with city/state | 40 | Null correlation |
| city | Correlated nulls with address | 40 | Null correlation |
| state | Correlated nulls with address | 40 | Null correlation |

- 15 columns WITHOUT planted issues (false positive traps):
  - **free_text_notes** — messy text, multiple formats, BUT intentionally correct
  - **product_description** — varying lengths, special characters, valid HTML entities
  - **currency_code** — 3-letter codes, low cardinality, NOT an error
  - **user_agent** — long strings, highly variable patterns, valid
  - **ip_address** — mix of IPv4 and IPv6, both valid
  - **tags** — comma-separated values, variable count, valid
  - **json_metadata** — serialized JSON strings, valid
  - **phone_intl** — intentionally mixed international formats (all valid)
  - **address_line2** — 60% null (legitimately optional, not an error)
  - **order_notes** — 80% null (optional field, not missing data)
  - **referral_code** — alphanumeric codes, looks random but valid
  - **session_id** — UUID format, all valid
  - **created_at** — timestamps with timezone, valid
  - **updated_at** — timestamps, always >= created_at, valid
  - **is_active** — boolean column (true/false strings), valid

Any tool that flags these clean columns loses precision points.

**Total planted issues:** ~500

## Tier 3: Adversarial (100K rows, 50 columns)

**Purpose:** Only the best tools (or LLM-enhanced tools) can handle this. Designed to expose limitations.

**Domain:** Healthcare claims + patient records (realistic sensitive data patterns).

**Columns (50):**
- 25 columns WITH planted issues (specific examples — full table defined in generator code):

| Column | Planted Issue | Count | Category |
|--------|--------------|-------|----------|
| npi_number | Valid 10-digit format but fails Luhn check digit | 50 | Semantic |
| patient_state | Valid 2-letter abbreviation but wrong for the zip | 30 | Cross-column |
| service_date | 20 dates before patient date_of_birth | 20 | Temporal |
| claim_notes | 15 Latin-1 characters (accented names) in UTF-8 column | 15 | Encoding |
| provider_name | 10 zero-width Unicode characters embedded | 10 | Encoding |
| diagnosis_desc | Smart quotes instead of straight quotes | 25 | Encoding |
| claim_amount | 12 values exceeding policy_max_amount | 12 | Cross-column |
| service_day | 20 weekend dates for weekday-only providers | 20 | Semantic |
| submit_date | 15 submitted before service_date | 15 | Temporal |
| record_number | 8 gaps in otherwise sequential numbering | 8 | Pattern |
| patient_age | 5 ages that don't match date_of_birth | 5 | Cross-column |
| procedure_code | 18 codes not in standard ICD-10 format | 18 | Format |
| insurance_id | 22 with wrong prefix for the listed provider_type | 22 | Cross-column |
| patient_zip | 10 zips that don't exist (valid format, fake location) | 10 | Semantic |
| dosage_amount | 8 extreme values (1000x typical) hidden in normal distribution | 8 | Statistical |
| lab_result | Bimodal: 2 patient populations, 6 values in the gap between modes | 6 | Statistical |
| admission_date | 12 after discharge_date | 12 | Temporal |
| discharge_date | Correlated temporal with admission | 12 | Temporal |
| primary_dx | 15 codes from deprecated ICD-9 mixed with ICD-10 | 15 | Format |
| secondary_dx | 10 codes that conflict with primary_dx | 10 | Cross-column |
| charge_code | 20 duplicates in sequential billing | 20 | Uniqueness |
| patient_name | 8 with numeric characters embedded ("J0hn Sm1th") | 8 | Encoding |
| referring_npi | 25 valid format but same Luhn failure as npi_number | 25 | Semantic |
| auth_number | 5 with wrong length for the insurance type | 5 | Cross-column |
| payment_amount | 7 negative values (refunds mislabeled as payments) | 7 | Semantic |

- 25 columns WITHOUT planted issues (harder false positive traps):
  - Columns with legitimate high null rates (optional fields)
  - Columns with legitimate bimodal distributions (two patient populations)
  - Columns with legitimate format variation (ICD-10 code versions)
  - Columns with legitimate outliers (neonatal vs geriatric age ranges)
  - Free-text fields with medical abbreviations that look like typos

**Total planted issues:** ~1000

## Scoring

### Per-Tier Metrics

For each tier, computed against ground truth:

```python
@dataclass
class TierResult:
    tier: int
    recall: float           # planted-issue columns detected / total planted-issue columns
    precision: float        # true positive columns / all flagged columns
    f1: float              # 2 * P * R / (P + R)
    false_positive_rate: float  # clean columns flagged / total clean columns
    time_seconds: float
    memory_mb: float
    findings_count: int
```

**Column-level evaluation:**
- A column counts as "detected" if any finding (any severity) targets it. This intentionally includes INFO-level findings — the benchmark rewards detection, and confidence/severity distinctions are captured by the optional calibration metric.
- A column counts as "false positive" if it has no planted issues but the tool flagged it with WARNING or ERROR severity. INFO-level findings on clean columns do NOT count as false positives (tools should be free to report observations without penalty).
- For comma-joined cross-column findings, each constituent column is evaluated separately

### DQBench Score

Single composite number (0-100):

```
DQBench Score = (Tier1_F1 × 0.20) + (Tier2_F1 × 0.40) + (Tier3_F1 × 0.40)
```

Tier 2 and 3 weighted equally and heavier than Tier 1 because harder tiers are better discriminators between tools — most tools will score well on Tier 1, so it carries less weight.

### Optional Metrics

Reported but not included in composite score:
- **LLM Cost** — total dollars spent on API calls (if applicable)
- **Confidence Calibration** — how well reported confidence predicts actual correctness. Computed as: bin findings by confidence (0-0.2, 0.2-0.4, etc.), compute actual precision per bin, report `1 - MAE` where MAE is mean absolute error vs. perfect calibration. Score of 1.0 = perfectly calibrated. Score of 0.0 = maximally miscalibrated.

### Scorecard Output

```
════════════════════════════════════════════════════════════
                    DQBench v1.0 Scorecard
════════════════════════════════════════════════════════════
Tool: GoldenCheck v0.2.0

TIER 1 — Basics (5K rows, 15 columns)
  Recall: 100%  Precision: 85%  F1: 92.0%
  FPR: 0%  Time: 0.12s  Memory: 8 MB  Findings: 45

TIER 2 — Realistic (50K rows, 30 columns)
  Recall: 88%   Precision: 72%  F1: 79.3%
  FPR: 18%  Time: 1.4s   Memory: 45 MB  Findings: 120

TIER 3 — Adversarial (100K rows, 50 columns)
  Recall: 65%   Precision: 60%  F1: 62.4%
  FPR: 22%  Time: 3.1s   Memory: 90 MB  Findings: 200

────────────────────────────────────────────────────────────
DQBench Score: 72.8 / 100

Optional:
  LLM Cost: $0.02
  Confidence Calibration: 0.85
════════════════════════════════════════════════════════════
```

Also saved as JSON for programmatic consumption.

## Ground Truth Format

Each tier's `ground_truth.json`:

```json
{
  "tier": 1,
  "version": "1.0",
  "rows": 5000,
  "columns": 15,
  "planted_columns": {
    "customer_id": {
      "issues": ["uniqueness"],
      "planted_count": 15,
      "description": "15 duplicate customer IDs",
      "affected_rows": [102, 305, ...]
    },
    "email": {
      "issues": ["format_detection"],
      "planted_count": 25,
      "description": "25 non-email values mixed in",
      "affected_rows": [...]
    }
  },
  "clean_columns": ["free_text_notes", "product_description", ...],
  "total_planted_issues": 200
}
```

The `clean_columns` list is what enables precision measurement — any finding on a clean column is a false positive.

## Dataset Generation

All datasets are generated deterministically with `random.seed(42)`. The generator uses realistic fake data:
- Names from a pool of common first/last names
- Emails matching name patterns
- Addresses with real city/state/zip combinations
- Dates within realistic ranges
- Domain-appropriate values (ICD-10 codes, insurance IDs, etc. for Tier 3)

Datasets are generated on first run and cached in `~/.dqbench/datasets/`. Regenerate with `dqbench generate --force`.

### Reproducibility Policy

- Generator uses only Python stdlib `random` (not numpy) for determinism
- Canonical datasets are committed to the repo as release artifacts (not generated ad-hoc)
- Any researcher can verify by running `dqbench generate` and comparing checksums
- Generator targets Python 3.11+ and documents the exact Python version used for canonical generation

### Version Stability Policy

- Once a benchmark version (e.g., v1.0) is published, its datasets and ground truth are immutable
- New versions (v1.1, v2.0) ship new datasets alongside old ones — old scores remain valid
- The `ground_truth.json` version field enables score comparison: only scores from the same version are comparable
- Breaking changes to scoring methodology require a major version bump

## Tech Stack

| Dependency | Purpose |
|-----------|---------|
| Polars | Dataset generation and CSV writing |
| Typer | CLI |
| Rich | Scorecard output |
| Pydantic 2 | Ground truth schema validation |

**No optional dependencies.** Adapters import the tool they're benchmarking — that's the tool's dependency, not dqbench's.

**Python 3.11+**

## Built-in Adapters

### GoldenCheck Adapter

```python
from dqbench.adapters.base import DQBenchAdapter, DQBenchFinding
from pathlib import Path

class GoldenCheckAdapter(DQBenchAdapter):
    @property
    def name(self) -> str:
        return "GoldenCheck"

    @property
    def version(self) -> str:
        from goldencheck import __version__
        return __version__

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        from goldencheck.engine.scanner import scan_file
        findings, _ = scan_file(csv_path)
        return [
            DQBenchFinding(
                column=f.column,
                severity=f.severity.name.lower(),
                check=f.check,
                message=f.message,
                confidence=f.confidence,
            )
            for f in findings
        ]
```

Other adapters (Great Expectations, Pandera, etc.) can be contributed by the community or added later.

## Repo Structure

Separate repo: `github.com/benzsevern/dqbench`

```
dqbench/
├── dqbench/            # Python package
├── tests/
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── docs/
```

## Out of Scope (v1.0)

- Built-in adapters for GX, Pandera, Pointblank (community contributions welcome)
- Leaderboard website
- Multi-file/multi-table benchmarks
- Streaming/real-time benchmarks
- Custom dataset upload
- Historical result tracking
