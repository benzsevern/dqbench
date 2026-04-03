"""ER Tier 1 dataset generator — 1,000 rows with 100 easy duplicate pairs."""
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
