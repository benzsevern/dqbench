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
        "case", "whitespace", "phone_format", "email_case",
        "extra_spaces", "unicode", "abbreviation",
    ])
    if issue_type == "case":
        field = rng.choice(["first_name", "last_name", "city", "company"])
        messy[field] = messy[field].lower() if rng.random() < 0.5 else messy[field].upper()
    elif issue_type == "whitespace":
        field = rng.choice(["first_name", "last_name", "company", "address"])
        messy[field] = f"  {messy[field]}  "
    elif issue_type == "phone_format":
        digits = "".join(c for c in messy["phone"] if c.isdigit())
        if len(digits) == 10:
            messy["phone"] = rng.choice([
                digits,
                f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
            ])
    elif issue_type == "email_case":
        messy["email"] = messy["email"].upper()
    elif issue_type == "extra_spaces":
        field = rng.choice(["address", "city", "company"])
        words = messy[field].split()
        messy[field] = "  ".join(words)
    elif issue_type == "unicode":
        field = rng.choice(["first_name", "last_name"])
        val = messy[field]
        if len(val) > 2:
            idx = rng.randint(1, len(val) - 1)
            messy[field] = val[:idx] + "\u200b" + val[idx:]
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
