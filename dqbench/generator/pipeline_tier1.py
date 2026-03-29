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
