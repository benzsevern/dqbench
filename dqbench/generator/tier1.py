"""Tier 1 dataset generator — 5,000-row customer database with planted issues."""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

import polars as pl

from dqbench.ground_truth import GroundTruth, PlantedColumn
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

NROWS = 5000


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def generate_tier1() -> tuple[pl.DataFrame, GroundTruth]:
    """Generate the Tier 1 benchmark dataset."""
    rng = random.Random(42)

    # ------------------------------------------------------------------ #
    # Pre-select which rows get which issues (all indices 0-based)        #
    # ------------------------------------------------------------------ #

    # customer_id: 15 duplicates
    dup_id_rows = rng.sample(range(NROWS), 15)

    # first_name: 8 nulls
    null_fname_rows = rng.sample(range(NROWS), 8)

    # last_name: 5 numeric strings
    numeric_lname_rows = rng.sample(range(NROWS), 5)

    # email: 25 non-email values
    bad_email_rows = rng.sample(range(NROWS), 25)
    bad_email_values = [
        "555-867-5309", "N/A", "n/a", "none", "NONE", "NULL", "not provided",
        "555-234-5678", "N/A", "gibberish@@", "plainaddress", "missing",
        "555-111-2222", "unknown", "@nodomain", "user@", "N/A",
        "555-999-0000", "no email", "@@@@", "1234567890", "N/A",
        "555-444-3333", "not available", "---",
    ]  # exactly 25

    # age: 20 string values + 8 outliers (28 distinct rows)
    age_str_pool = ["thirty", "twenty-five", "forty", "fifty", "sixty", "eighteen",
                    "twenty", "thirty-five", "forty-five", "fifty-five",
                    "sixty-five", "seventy", "nineteen", "twenty-two", "twenty-eight",
                    "thirty-two", "forty-two", "fifty-two", "sixty-two", "seventy-five"]
    age_outlier_pool = [999, -1, 0, 200, 999, -1, 0, 200]
    age_str_rows = rng.sample(range(NROWS), 20)
    remaining_for_outlier = [i for i in range(NROWS) if i not in age_str_rows]
    age_outlier_rows = rng.sample(remaining_for_outlier, 8)

    # income: 10 extreme outliers
    income_outlier_rows = rng.sample(range(NROWS), 10)

    # status: 15 misspelled
    misspelled_statuses = ["actve", "ACTIVE", "Inactive", "pendng", "actve",
                           "ACTIVE", "Inactive", "pendng", "actve", "ACTIVE",
                           "Inactive", "pendng", "actve", "ACTIVE", "Inactive"]
    status_bad_rows = rng.sample(range(NROWS), 15)

    # signup_date: 12 wrong format (MM-DD-YYYY)
    date_fmt_rows = rng.sample(range(NROWS), 12)

    # last_login: 18 where last_login < signup_date
    login_before_signup_rows = rng.sample(range(NROWS), 18)

    # country: 10 invalid codes
    invalid_country_pool = ["XX", "ZZ", "QQ", "99", "XX", "ZZ", "QQ", "99", "XX", "ZZ"]
    country_bad_rows = rng.sample(range(NROWS), 10)

    # shipping_address / city / zip: 50 correlated nulls
    null_shipping_rows = set(rng.sample(range(NROWS), 50))

    # ------------------------------------------------------------------ #
    # Build columns row by row                                            #
    # ------------------------------------------------------------------ #

    start_date = date(2020, 1, 1)
    end_date = date(2024, 12, 31)

    # Base sequential IDs
    customer_ids: list[int | str] = list(range(1, NROWS + 1))
    # Plant duplicates: replace dup_id_rows[i] value with a randomly chosen existing id
    # pick 15 source rows whose ids we will duplicate (not in dup_id_rows)
    dup_source_rows = rng.sample([r for r in range(NROWS) if r not in dup_id_rows], 15)
    for target_row, source_row in zip(dup_id_rows, dup_source_rows):
        customer_ids[target_row] = customer_ids[source_row]

    first_names: list[str | None] = [rng.choice(FIRST_NAMES) for _ in range(NROWS)]
    for r in null_fname_rows:
        first_names[r] = None

    last_names: list[str] = [rng.choice(LAST_NAMES) for _ in range(NROWS)]
    numeric_strings = ["12345", "67890", "11111", "54321", "99999"]
    for i, r in enumerate(numeric_lname_rows):
        last_names[r] = numeric_strings[i]

    # Email — build base from name + domain
    emails: list[str] = []
    for i in range(NROWS):
        fn = first_names[i] if first_names[i] is not None else rng.choice(FIRST_NAMES)
        ln = last_names[i]
        domain = rng.choice(DOMAINS)
        emails.append(f"{fn.lower()}.{ln.lower()}@{domain}")
    for i, r in enumerate(bad_email_rows):
        emails[r] = bad_email_values[i]

    # Phone — 3 formats distributed across all rows
    phones: list[str] = []
    for i in range(NROWS):
        area = rng.randint(200, 999)
        prefix = rng.randint(200, 999)
        line = rng.randint(1000, 9999)
        fmt = rng.choice([0, 1, 2])
        if fmt == 0:
            phones.append(f"({area}) {prefix}-{line}")
        elif fmt == 1:
            phones.append(f"{area}-{prefix}-{line}")
        else:
            phones.append(f"{area}{prefix}{line}")

    # Age
    ages: list[int | str] = [rng.randint(18, 80) for _ in range(NROWS)]
    for i, r in enumerate(age_str_rows):
        ages[r] = age_str_pool[i]
    for i, r in enumerate(age_outlier_rows):
        ages[r] = age_outlier_pool[i]

    # Income
    incomes: list[float] = [round(rng.uniform(25000, 150000), 2) for _ in range(NROWS)]
    for r in income_outlier_rows:
        incomes[r] = 9999999.99

    # Status
    statuses: list[str] = [rng.choice(STATUSES) for _ in range(NROWS)]
    for i, r in enumerate(status_bad_rows):
        statuses[r] = misspelled_statuses[i]

    # Signup date + last_login
    signup_dates_raw: list[date] = [_rand_date(rng, start_date, end_date) for _ in range(NROWS)]
    last_logins_raw: list[date] = [
        d + timedelta(days=rng.randint(1, 365)) for d in signup_dates_raw
    ]
    # Plant login before signup
    for r in login_before_signup_rows:
        sd = signup_dates_raw[r]
        # pick a date 1-30 days BEFORE signup
        last_logins_raw[r] = sd - timedelta(days=rng.randint(1, 30))

    # Format signup_date (12 rows use MM-DD-YYYY)
    date_fmt_set = set(date_fmt_rows)
    signup_dates: list[str] = []
    for i, d in enumerate(signup_dates_raw):
        if i in date_fmt_set:
            signup_dates.append(d.strftime("%m-%d-%Y"))
        else:
            signup_dates.append(_fmt_date(d))
    last_logins: list[str] = [_fmt_date(d) for d in last_logins_raw]

    # Country
    countries: list[str] = [rng.choice(COUNTRIES) for _ in range(NROWS)]
    for i, r in enumerate(country_bad_rows):
        countries[r] = invalid_country_pool[i]

    # Zip code — mix 5-digit and 9-digit (XXXXX-XXXX)
    zip_codes: list[str] = []
    for _ in range(NROWS):
        base = str(rng.randint(10000, 99999))
        if rng.random() < 0.4:
            ext = str(rng.randint(1000, 9999))
            zip_codes.append(f"{base}-{ext}")
        else:
            zip_codes.append(base)

    # Shipping address / city / zip — 50 correlated nulls
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

    # ------------------------------------------------------------------ #
    # Clean columns                                                       #
    # ------------------------------------------------------------------ #

    order_counts: list[int] = [rng.randint(0, 50) for _ in range(NROWS)]
    account_types: list[str] = [rng.choice(ACCOUNT_TYPES) for _ in range(NROWS)]

    # last_updated: datetime after signup_date, with UTC timezone
    last_updated: list[str] = []
    for i, d in enumerate(signup_dates_raw):
        days_after = rng.randint(0, 365)
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=days_after)
        last_updated.append(dt.isoformat())

    # notes: 70% null, free text otherwise
    notes: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.70:
            notes.append(None)
        else:
            notes.append(rng.choice(FREE_TEXT_NOTES))

    referral_sources: list[str] = [rng.choice(REFERRAL_SOURCES) for _ in range(NROWS)]

    # ------------------------------------------------------------------ #
    # Build Polars DataFrame                                              #
    # ------------------------------------------------------------------ #

    # ages and customer_ids contain mixed types; cast to string for those columns
    age_strs: list[str | None] = [str(a) for a in ages]
    cid_strs: list[str] = [str(c) for c in customer_ids]

    df = pl.DataFrame(
        {
            "customer_id": cid_strs,
            "first_name": first_names,
            "last_name": last_names,
            "email": emails,
            "phone": phones,
            "age": age_strs,
            "income": incomes,
            "status": statuses,
            "signup_date": signup_dates,
            "last_login": last_logins,
            "country": countries,
            "zip_code": zip_codes,
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

    # ------------------------------------------------------------------ #
    # Build GroundTruth                                                   #
    # ------------------------------------------------------------------ #

    planted: dict[str, PlantedColumn] = {
        "customer_id": PlantedColumn(
            issues=["duplicate_values"],
            planted_count=15,
            description="15 rows have duplicate customer IDs (originally unique sequential IDs).",
            affected_rows=sorted(dup_id_rows),
        ),
        "first_name": PlantedColumn(
            issues=["null_values"],
            planted_count=8,
            description="8 rows have null first_name values.",
            affected_rows=sorted(null_fname_rows),
        ),
        "last_name": PlantedColumn(
            issues=["wrong_type"],
            planted_count=5,
            description="5 rows have numeric strings instead of name strings.",
            affected_rows=sorted(numeric_lname_rows),
        ),
        "email": PlantedColumn(
            issues=["invalid_format"],
            planted_count=25,
            description="25 rows contain non-email values (phone numbers, 'N/A', gibberish).",
            affected_rows=sorted(bad_email_rows),
        ),
        "phone": PlantedColumn(
            issues=["inconsistent_format"],
            planted_count=NROWS,
            description="Phone numbers use 3 different formats across all rows: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXXXXXXXXX.",
            affected_rows=list(range(NROWS)),
        ),
        "age": PlantedColumn(
            issues=["wrong_type", "outlier_values"],
            planted_count=28,
            description="20 rows have string age values (e.g. 'thirty'); 8 rows have outlier values (999, -1, 0, 200).",
            affected_rows=sorted(age_str_rows + age_outlier_rows),
        ),
        "income": PlantedColumn(
            issues=["outlier_values"],
            planted_count=10,
            description="10 rows have extreme outlier income value 9999999.99.",
            affected_rows=sorted(income_outlier_rows),
        ),
        "status": PlantedColumn(
            issues=["misspelled_values"],
            planted_count=15,
            description="15 rows have misspelled status values (e.g. 'actve', 'ACTIVE', 'pendng').",
            affected_rows=sorted(status_bad_rows),
        ),
        "signup_date": PlantedColumn(
            issues=["inconsistent_format"],
            planted_count=12,
            description="12 rows use MM-DD-YYYY format instead of YYYY-MM-DD.",
            affected_rows=sorted(date_fmt_rows),
        ),
        "last_login": PlantedColumn(
            issues=["logic_violation"],
            planted_count=18,
            description="18 rows have last_login date before signup_date.",
            affected_rows=sorted(login_before_signup_rows),
        ),
        "country": PlantedColumn(
            issues=["invalid_values"],
            planted_count=10,
            description="10 rows have invalid country codes ('XX', 'ZZ', 'QQ', '99').",
            affected_rows=sorted(country_bad_rows),
        ),
        "zip_code": PlantedColumn(
            issues=["inconsistent_format"],
            planted_count=NROWS,
            description="Zip codes mix 5-digit and 9-digit (XXXXX-XXXX) formats across all rows.",
            affected_rows=list(range(NROWS)),
        ),
        "shipping_address": PlantedColumn(
            issues=["null_values"],
            planted_count=50,
            description="50 rows have null shipping_address (correlated with shipping_city and shipping_zip nulls).",
            affected_rows=sorted(null_shipping_rows),
        ),
        "shipping_city": PlantedColumn(
            issues=["null_values"],
            planted_count=50,
            description="50 rows have null shipping_city (correlated with shipping_address and shipping_zip nulls).",
            affected_rows=sorted(null_shipping_rows),
        ),
        "shipping_zip": PlantedColumn(
            issues=["null_values"],
            planted_count=50,
            description="50 rows have null shipping_zip (correlated with shipping_address and shipping_city nulls).",
            affected_rows=sorted(null_shipping_rows),
        ),
    }

    clean = ["order_count", "account_type", "last_updated", "notes", "referral_source"]

    total = sum(p.planted_count for p in planted.values())

    gt = GroundTruth(
        tier=1,
        version="1.0.0",
        rows=NROWS,
        columns=20,
        planted_columns=planted,
        clean_columns=clean,
        total_planted_issues=total,
    )

    return df, gt
