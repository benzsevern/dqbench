"""Generate clean ground truth CSVs for transform benchmarks."""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
    COUNTRIES,
    STATUSES,
    ACCOUNT_TYPES,
    REFERRAL_SOURCES,
    FREE_TEXT_NOTES,
)

# Issue types that have a deterministic correct transformation
TRANSFORMABLE_ISSUES = {
    "inconsistent_format",    # phone formats, date formats, zip formats
    "invalid_format",         # bad emails (clean = restore original email)
    "misspelled_values",      # status misspellings (clean = canonical)
    "wrong_type",             # age as string, numeric last names
    "invalid_values",         # bad country codes
}

# Issue types that are detection-only (no deterministic clean value)
DETECTION_ONLY_ISSUES = {
    "duplicate_values",       # no single correct resolution
    "outlier_values",         # no ground truth for correct value
    "logic_violation",        # ambiguous which date to fix
    "null_values",            # no original value to restore
}

_T1_NROWS = 5_000
_T2_NROWS = 50_000
_T3_NROWS = 100_000


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def generate_clean_tier1() -> pl.DataFrame:
    """Replay the tier1 RNG to produce a clean (ground-truth) DataFrame.

    Transformable columns are normalised to their canonical form.
    Detection-only columns keep the messy value (they won't be scored).
    """
    rng = random.Random(42)
    NROWS = _T1_NROWS

    # ------------------------------------------------------------------ #
    # Replay ALL rng draws in exactly the same order as tier1.py so that  #
    # every downstream draw is consistent.                                 #
    # ------------------------------------------------------------------ #

    # Pre-select rows (same order as tier1.py)
    dup_id_rows = rng.sample(range(NROWS), 15)
    null_fname_rows = rng.sample(range(NROWS), 8)
    numeric_lname_rows = rng.sample(range(NROWS), 5)
    bad_email_rows = rng.sample(range(NROWS), 25)

    age_str_rows = rng.sample(range(NROWS), 20)
    remaining_for_outlier = [i for i in range(NROWS) if i not in age_str_rows]
    age_outlier_rows = rng.sample(remaining_for_outlier, 8)

    income_outlier_rows = rng.sample(range(NROWS), 10)
    status_bad_rows = rng.sample(range(NROWS), 15)
    date_fmt_rows = rng.sample(range(NROWS), 12)
    login_before_signup_rows = rng.sample(range(NROWS), 18)
    country_bad_rows = rng.sample(range(NROWS), 10)
    null_shipping_rows = set(rng.sample(range(NROWS), 50))

    # ------------------------------------------------------------------ #
    # Replay column generation                                            #
    # ------------------------------------------------------------------ #

    start_date = date(2020, 1, 1)
    end_date = date(2024, 12, 31)

    # customer_id — keep messy (detection-only: duplicates)
    customer_ids: list[int | str] = list(range(1, NROWS + 1))
    dup_source_rows = rng.sample([r for r in range(NROWS) if r not in dup_id_rows], 15)
    for target_row, source_row in zip(dup_id_rows, dup_source_rows):
        customer_ids[target_row] = customer_ids[source_row]  # messy value kept

    # first_name — keep messy (detection-only: null_values)
    first_names: list[str | None] = [rng.choice(FIRST_NAMES) for _ in range(NROWS)]
    for r in null_fname_rows:
        first_names[r] = None  # messy value kept

    # last_name — TRANSFORMABLE: wrong_type (numeric strings → name strings)
    # Clean version = the originally-generated name string before mutation.
    # We regenerated the name already via rng.choice(LAST_NAMES); we just
    # don't apply the numeric_strings override.
    last_names: list[str] = [rng.choice(LAST_NAMES) for _ in range(NROWS)]
    # Do NOT apply numeric_strings mutation → clean version kept.
    # But we must still consume the pool iteration the same way tier1 does
    # (tier1 iterates numeric_lname_rows in enumerate order — no extra rng draws).

    # email — TRANSFORMABLE: invalid_format (non-email → original generated email)
    # The clean email is the generated fn.ln@domain value.
    # We must consume the rng.choice(FIRST_NAMES) call for null first_name rows
    # to stay in sync with tier1's email loop.
    emails_clean: list[str] = []
    for i in range(NROWS):
        fn = first_names[i] if first_names[i] is not None else rng.choice(FIRST_NAMES)
        ln = last_names[i]
        domain = rng.choice(DOMAINS)
        emails_clean.append(f"{fn.lower()}.{ln.lower()}@{domain}")
    # Do NOT apply bad_email_values overrides → clean email is the original generated one.

    # phone — TRANSFORMABLE: inconsistent_format → canonical (XXX) XXX-XXXX
    phones_clean: list[str] = []
    for i in range(NROWS):
        area = rng.randint(200, 999)
        prefix = rng.randint(200, 999)
        line = rng.randint(1000, 9999)
        _fmt = rng.choice([0, 1, 2])  # consume the rng draw (same as tier1)
        # Clean canonical format is always fmt==0: (XXX) XXX-XXXX
        phones_clean.append(f"({area}) {prefix}-{line}")

    # age — TRANSFORMABLE for wrong_type (string ages), DETECTION-ONLY for outliers.
    # Clean: restore the integer for string-type rows; keep messy for outliers.
    ages_base: list[int] = [rng.randint(18, 80) for _ in range(NROWS)]
    ages_clean: list[str | None] = [str(a) for a in ages_base]
    # For age_str_rows: the original value is already in ages_base (integer).
    # Clean version is the integer string — same as ages_base[r].
    # Nothing to override for those rows.
    # For age_outlier_rows: detection-only → keep the messy value (outlier number).
    age_outlier_pool = [999, -1, 0, 200, 999, -1, 0, 200]
    for i, r in enumerate(age_outlier_rows):
        ages_clean[r] = str(age_outlier_pool[i])  # keep messy (detection-only)

    # income — keep messy (detection-only: outlier_values)
    incomes: list[float] = [round(rng.uniform(25000, 150000), 2) for _ in range(NROWS)]
    for r in income_outlier_rows:
        incomes[r] = 9999999.99  # keep messy value

    # status — TRANSFORMABLE: misspelled_values → canonical
    misspelled_statuses = [
        "actve", "ACTIVE", "Inactive", "pendng", "actve",
        "ACTIVE", "Inactive", "pendng", "actve", "ACTIVE",
        "Inactive", "pendng", "actve", "ACTIVE", "Inactive",
    ]
    # Canonical mapping (case-insensitive)
    _status_canon = {
        "actve": "active", "active": "active", "ACTIVE": "active",
        "inactive": "inactive", "Inactive": "inactive", "INACTIVE": "inactive",
        "pendng": "pending", "pending": "pending", "PENDING": "pending",
    }
    statuses_base: list[str] = [rng.choice(STATUSES) for _ in range(NROWS)]
    statuses_clean: list[str] = list(statuses_base)
    for i, r in enumerate(status_bad_rows):
        bad_val = misspelled_statuses[i]
        statuses_clean[r] = _status_canon.get(bad_val, bad_val)

    # signup_date — TRANSFORMABLE: inconsistent_format → YYYY-MM-DD
    signup_dates_raw: list[date] = [_rand_date(rng, start_date, end_date) for _ in range(NROWS)]
    signup_dates_clean: list[str] = [_fmt_date(d) for d in signup_dates_raw]
    # All rows normalised to ISO format (date_fmt_rows had MM-DD-YYYY → now YYYY-MM-DD).

    # last_login — consume rng draws (same as tier1), keep messy (detection-only)
    last_logins_raw: list[date] = [
        d + timedelta(days=rng.randint(1, 365)) for d in signup_dates_raw
    ]
    for r in login_before_signup_rows:
        sd = signup_dates_raw[r]
        last_logins_raw[r] = sd - timedelta(days=rng.randint(1, 30))
    last_logins_clean: list[str] = [_fmt_date(d) for d in last_logins_raw]

    # country — TRANSFORMABLE: invalid_values → drop invalid codes (keep messy for now,
    # since we don't know the "correct" country; detection-only in practice)
    # Per spec: invalid_values IS transformable, but for country we have no correct
    # value to restore (the original was already a valid country from COUNTRIES pool).
    # Actually, the country column IS generated from COUNTRIES first, THEN overwritten.
    # We can restore the original generated value for the bad rows.
    countries_base: list[str] = [rng.choice(COUNTRIES) for _ in range(NROWS)]
    countries_clean: list[str] = list(countries_base)
    invalid_country_pool = ["XX", "ZZ", "QQ", "99", "XX", "ZZ", "QQ", "99", "XX", "ZZ"]
    # For country_bad_rows, countries_base already has the original valid value.
    # The messy CSV overwrites them with invalid codes.
    # Clean version = countries_base[r] (original valid country code).
    # countries_clean is already set to countries_base, so nothing to change.

    # zip_code — TRANSFORMABLE: inconsistent_format → 5-digit only
    zip_codes_clean: list[str] = []
    for _ in range(NROWS):
        base = str(rng.randint(10000, 99999))
        if rng.random() < 0.4:
            _ext = str(rng.randint(1000, 9999))  # consume rng but discard extension
            zip_codes_clean.append(base)
        else:
            zip_codes_clean.append(base)

    # shipping_address / city / zip — keep messy (detection-only: null_values)
    shipping_addresses: list[str | None] = []
    shipping_cities: list[str | None] = []
    shipping_zips: list[str | None] = []
    street_suffixes = ["Main St", "Oak Ave", "Elm Rd", "Maple Dr", "Cedar Ln",
                       "Pine Blvd", "Park Way", "Lake Dr", "Hill Rd", "River Rd"]
    for i in range(NROWS):
        if i in null_shipping_rows:
            shipping_addresses.append(None)
            shipping_cities.append(None)
            shipping_zips.append(None)
        else:
            num = rng.randint(1, 9999)
            suffix = rng.choice(street_suffixes)
            shipping_addresses.append(f"{num} {suffix}")
            city_name, _state, city_zip = rng.choice(CITIES)
            shipping_cities.append(city_name)
            shipping_zips.append(city_zip)

    # Clean columns (no issues — same as messy)
    order_counts: list[int] = [rng.randint(0, 50) for _ in range(NROWS)]
    account_types: list[str] = [rng.choice(ACCOUNT_TYPES) for _ in range(NROWS)]

    from datetime import datetime, timezone
    last_updated: list[str] = []
    for i, d in enumerate(signup_dates_raw):
        days_after = rng.randint(0, 365)
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=days_after)
        last_updated.append(dt.isoformat())

    notes: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.70:
            notes.append(None)
        else:
            notes.append(rng.choice(FREE_TEXT_NOTES))

    referral_sources: list[str] = [rng.choice(REFERRAL_SOURCES) for _ in range(NROWS)]

    cid_strs: list[str] = [str(c) for c in customer_ids]

    df = pl.DataFrame(
        {
            "customer_id": cid_strs,
            "first_name": first_names,
            "last_name": last_names,
            "email": emails_clean,
            "phone": phones_clean,
            "age": ages_clean,
            "income": incomes,
            "status": statuses_clean,
            "signup_date": signup_dates_clean,
            "last_login": last_logins_clean,
            "country": countries_clean,
            "zip_code": zip_codes_clean,
            "shipping_address": shipping_addresses,
            "shipping_city": shipping_cities,
            "shipping_zip": shipping_zips,
            "order_count": order_counts,
            "account_type": account_types,
            "last_updated": last_updated,
            "notes": notes,
            "referral_source": referral_sources,
        }
    )
    return df


def generate_clean_tier2() -> pl.DataFrame:
    """Stub: copy the messy tier2 data (full clean generation TBD)."""
    from dqbench.runner import CACHE_DIR
    return pl.read_csv(CACHE_DIR / "tier2" / "data.csv", infer_schema_length=0)


def generate_clean_tier3() -> pl.DataFrame:
    """Stub: copy the messy tier3 data (full clean generation TBD)."""
    from dqbench.runner import CACHE_DIR
    return pl.read_csv(CACHE_DIR / "tier3" / "data.csv", infer_schema_length=0)


def generate_clean_csvs(cache_dir: Path) -> None:
    """Generate clean ground truth CSVs for all three tiers."""
    # Tier 1 — full clean generation
    clean1 = generate_clean_tier1()
    clean1.write_csv(cache_dir / "tier1" / "data_clean.csv")

    # Tier 2 — stub (copy messy)
    t2_path = cache_dir / "tier2" / "data.csv"
    if t2_path.exists():
        df2 = pl.read_csv(t2_path, infer_schema_length=0)
        df2.write_csv(cache_dir / "tier2" / "data_clean.csv")

    # Tier 3 — stub (copy messy)
    t3_path = cache_dir / "tier3" / "data.csv"
    if t3_path.exists():
        df3 = pl.read_csv(t3_path, infer_schema_length=0)
        df3.write_csv(cache_dir / "tier3" / "data_clean.csv")
