"""Deterministic OCR-company benchmark generator."""
from __future__ import annotations

import random

import polars as pl

BASE_COMPANY_NAMES = [
    "Atlas Equipment Rental LLC",
    "Blue Ridge Machinery Inc",
    "Iron Summit Services LLC",
    "North Valley Contractors LLC",
    "Cedar Rock Aggregates Inc",
    "Frontier Fleet Solutions LLC",
    "Titan Crane & Rigging LLC",
    "Granite State Equipment Co",
    "Riverbend Diesel Repair LLC",
    "Pioneer Site Services Inc",
    "Highline Utility Construction LLC",
    "Keystone Earthmoving LLC",
    "Summit Power Generation LLC",
    "Metro Pump and Compressor Inc",
    "Red Oak Hauling LLC",
    "Silver Peak Industrial Services LLC",
    "Longhorn Materials Handling LLC",
    "American Field Mechanics LLC",
    "Northstar Parts Depot Inc",
    "Bulldog Equipment Finance LLC",
]

DOCUMENT_TYPES = ["invoice", "service_order", "equipment_listing", "rental_agreement"]
CONFUSIONS = [("O", "0"), ("I", "1"), ("L", "I"), ("S", "5"), ("B", "8"), ("M", "RN")]


def generate_ocr_company_tier(tier: int) -> pl.DataFrame:
    seeds = {1: 4101, 2: 4102, 3: 4103}
    sizes = {1: 120, 2: 150, 3: 180}
    corrupted_probability = {1: 0.45, 2: 0.50, 3: 0.55}[tier]
    rng = random.Random(seeds[tier])

    records: list[dict[str, object]] = []
    for index in range(sizes[tier]):
        truth = BASE_COMPANY_NAMES[index % len(BASE_COMPANY_NAMES)]
        if index >= len(BASE_COMPANY_NAMES):
            truth = vary_company_name(truth, rng)
        corrupted = rng.random() < corrupted_probability
        ocr_name = inject_tier_noise(truth.upper(), tier, rng) if corrupted else truth.upper()
        records.append(
            {
                "record_id": f"OCR-{tier}-{index:04d}",
                "document_type": rng.choice(DOCUMENT_TYPES),
                "company_name_truth": truth.upper(),
                "company_name_ocr": ocr_name,
                "company_corrupted": corrupted,
            }
        )
    return pl.DataFrame(records)


def vary_company_name(name: str, rng: random.Random) -> str:
    prefixes = ["Atlas", "Summit", "Blue", "Granite", "Metro", "True", "Pioneer", "Beacon"]
    modifiers = ["Field", "Industrial", "Heavy", "Utility", "Fleet", "Equipment", "Site", "Service"]
    parts = name.split()
    parts[0] = rng.choice(prefixes)
    if len(parts) > 2:
        parts[1] = rng.choice(modifiers)
    return " ".join(parts)


def inject_tier_noise(value: str, tier: int, rng: random.Random) -> str:
    corrupted = inject_simple_noise(value, rng)
    if tier >= 2:
        corrupted = inject_token_boundary_noise(corrupted, rng)
    if tier >= 3:
        corrupted = inject_adversarial_noise(corrupted, rng)
    return corrupted


def inject_simple_noise(value: str, rng: random.Random) -> str:
    corrupted = value
    for source, target in rng.sample(CONFUSIONS, k=2):
        if source in corrupted and rng.random() < 0.7:
            corrupted = corrupted.replace(source, target, 1)
    tokens = corrupted.split()
    if tokens and rng.random() < 0.4:
        index = rng.randrange(len(tokens))
        token = tokens[index]
        if len(token) > 4:
            char_index = rng.randrange(1, len(token) - 1)
            tokens[index] = token[:char_index] + token[char_index + 1:]
            corrupted = " ".join(tokens)
    return corrupted


def inject_token_boundary_noise(value: str, rng: random.Random) -> str:
    tokens = value.split()
    if len(tokens) >= 2 and rng.random() < 0.55:
        index = rng.randrange(len(tokens) - 1)
        tokens[index:index + 2] = [tokens[index] + tokens[index + 1]]
    elif tokens and rng.random() < 0.35:
        index = rng.randrange(len(tokens))
        token = tokens[index]
        if len(token) > 6:
            split_at = len(token) // 2
            tokens[index:index + 1] = [token[:split_at], token[split_at:]]
    return " ".join(tokens)


def inject_adversarial_noise(value: str, rng: random.Random) -> str:
    corrupted = value
    if "LLC" in corrupted and rng.random() < 0.7:
        corrupted = corrupted.replace("LLC", "ILC", 1)
    elif "INC" in corrupted and rng.random() < 0.5:
        corrupted = corrupted.replace("INC", "1NC", 1)
    if rng.random() < 0.35:
        corrupted = corrupted.replace(" & ", "&", 1)
    return corrupted
