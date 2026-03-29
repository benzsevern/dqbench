"""ER Tier 3 dataset generator — 10,000 rows with 2,000 adversarial duplicate pairs."""
from __future__ import annotations

import random

import polars as pl

from dqbench.er_ground_truth import ERGroundTruth
from dqbench.generator.er_tier1 import _generate_entity, _generate_phone, _generate_address
from dqbench.generator.utils import (
    NICKNAME_MAP,
    PHONETIC_VARIANTS,
    ADDRESS_ABBREVIATIONS,
)

NROWS = 10000
N_UNIQUE = 8000
N_DUPES = 2000


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
