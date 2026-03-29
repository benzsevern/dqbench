"""ER Tier 2 dataset generator — 5,000 rows with 750 fuzzy duplicate pairs."""
from __future__ import annotations

import random

import polars as pl

from dqbench.er_ground_truth import ERGroundTruth
from dqbench.generator.er_tier1 import _generate_entity, _generate_phone, _generate_address
from dqbench.generator.utils import NICKNAME_MAP

NROWS = 5000
N_UNIQUE = 4250
N_DUPES = 750


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
