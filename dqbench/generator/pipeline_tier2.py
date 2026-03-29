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
