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
    _date_fmt_rows = rng.sample(range(NROWS), 12)  # noqa: F841
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

    # last_name — DETECTION-ONLY: wrong_type (numeric strings — no correct transform)
    # Clean version = same as messy (detection-only, no auto-correction possible).
    last_names: list[str] = [rng.choice(LAST_NAMES) for _ in range(NROWS)]
    # Apply the same numeric_strings mutation as tier1 so messy == clean for these rows.
    numeric_strings = ["12345", "67890", "11111", "54321", "99999"]
    for i, r in enumerate(numeric_lname_rows):
        last_names[r] = numeric_strings[i]

    # email — DETECTION-ONLY: invalid_format (non-email values like "N/A", "555-867-5309")
    # A transform tool cannot restore the original email from garbage input.
    # Clean email = generated value, but bad_email_rows get the messy value (same as tier1).
    bad_email_values = [
        "555-867-5309", "N/A", "n/a", "none", "NONE", "NULL", "not provided",
        "555-234-5678", "N/A", "gibberish@@", "plainaddress", "missing",
        "555-111-2222", "unknown", "@nodomain", "user@", "N/A",
        "555-999-0000", "no email", "@@@@", "1234567890", "N/A",
        "555-444-3333", "not available", "---",
    ]
    emails_clean: list[str] = []
    for i in range(NROWS):
        fn = first_names[i] if first_names[i] is not None else rng.choice(FIRST_NAMES)
        ln = last_names[i]
        domain = rng.choice(DOMAINS)
        emails_clean.append(f"{fn.lower()}.{ln.lower()}@{domain}")
    # Apply bad_email_values to match messy (detection-only — no correct transform exists)
    for i, r in enumerate(bad_email_rows):
        emails_clean[r] = bad_email_values[i]

    # phone — TRANSFORMABLE: inconsistent_format → E.164 canonical (+1XXXXXXXXXX)
    phones_clean: list[str] = []
    for i in range(NROWS):
        area = rng.randint(200, 999)
        prefix = rng.randint(200, 999)
        line = rng.randint(1000, 9999)
        _fmt = rng.choice([0, 1, 2])  # consume the rng draw (same as tier1)
        # Clean canonical format is E.164: +1{area}{prefix}{line}
        phones_clean.append(f"+1{area}{prefix}{line}")

    # age — DETECTION-ONLY for both wrong_type (string ages) and outliers.
    # Clean version = same as messy for all age issues (no auto-correction).
    age_str_pool = ["thirty", "twenty-five", "forty", "fifty", "sixty", "eighteen",
                    "twenty", "thirty-five", "forty-five", "fifty-five",
                    "sixty-five", "seventy", "nineteen", "twenty-two", "twenty-eight",
                    "thirty-two", "forty-two", "fifty-two", "sixty-two", "seventy-five"]
    ages_base: list[int] = [rng.randint(18, 80) for _ in range(NROWS)]
    ages_clean: list[str | None] = [str(a) for a in ages_base]
    # Apply string age mutation (same as tier1) — detection-only, keep messy value.
    for i, r in enumerate(age_str_rows):
        ages_clean[r] = age_str_pool[i]
    # Apply outlier mutation (same as tier1) — detection-only, keep messy value.
    age_outlier_pool = [999, -1, 0, 200, 999, -1, 0, 200]
    for i, r in enumerate(age_outlier_rows):
        ages_clean[r] = str(age_outlier_pool[i])  # keep messy (detection-only)

    # income — keep messy (detection-only: outlier_values)
    incomes: list[float] = [round(rng.uniform(25000, 150000), 2) for _ in range(NROWS)]
    for r in income_outlier_rows:
        incomes[r] = 9999999.99  # keep messy value

    # status — DETECTION-ONLY: misspelled_values (no auto-correction in GoldenFlow yet)
    # Clean version = same as messy (keep misspelled values).
    misspelled_statuses = [
        "actve", "ACTIVE", "Inactive", "pendng", "actve",
        "ACTIVE", "Inactive", "pendng", "actve", "ACTIVE",
        "Inactive", "pendng", "actve", "ACTIVE", "Inactive",
    ]
    statuses_base: list[str] = [rng.choice(STATUSES) for _ in range(NROWS)]
    statuses_clean: list[str] = list(statuses_base)
    # Apply the same misspelling mutation as tier1 so messy == clean for these rows.
    for i, r in enumerate(status_bad_rows):
        statuses_clean[r] = misspelled_statuses[i]

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

    # country — DETECTION-ONLY: invalid_values (GoldenFlow doesn't auto-detect these yet)
    # Clean version = same as messy (keep invalid codes).
    countries_base: list[str] = [rng.choice(COUNTRIES) for _ in range(NROWS)]
    countries_clean: list[str] = list(countries_base)
    invalid_country_pool = ["XX", "ZZ", "QQ", "99", "XX", "ZZ", "QQ", "99", "XX", "ZZ"]
    # Apply the same invalid country mutation as tier1 so messy == clean for these rows.
    for i, r in enumerate(country_bad_rows):
        countries_clean[r] = invalid_country_pool[i]

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
    """Replay the tier2 RNG to produce a clean (ground-truth) DataFrame.

    Transformable columns are normalised to their canonical form.
    Detection-only columns keep the messy value (they won't be scored).

    Transformable issues in tier 2:
    - billing_zip: stored as Int64 (loses leading zeros) → zero-pad to 5-digit string

    Detection-only (kept as messy):
    - order_total: outlier_values
    - customer_email: invalid_format (garbage values — no original to restore)
    - product_category: distribution_shift
    - website_url: wrong_type (email values — no URL to restore)
    - phone_number: invalid_format (letters in area code — can't recover digits)
    - ship_date: logic_violation
    - quantity: invalid_values (negatives)
    - discount_pct: out_of_range
    - customer_name: wrong_type (numeric chars — no correction possible)
    - sku: duplicate_values
    - rating: enum_violation
    - address_line1/city/state: null_values
    """
    rng = random.Random(42)
    NROWS = _T2_NROWS

    # ------------------------------------------------------------------ #
    # Pre-select issue rows (same order as tier2.py)                      #
    # ------------------------------------------------------------------ #

    # 1. order_total: 20 outliers
    outlier_total_rows = rng.sample(range(NROWS), 20)
    outlier_total_set = set(outlier_total_rows)

    # 2. customer_email: 200 invalid
    bad_email_rows = rng.sample(range(NROWS), 200)
    bad_email_set = set(bad_email_rows)

    # 3. product_category: drift (last 10K) — no rng draws needed

    # 4. website_url: 15 email-format values
    bad_url_rows = rng.sample(range(NROWS), 15)
    bad_url_sorted = sorted(bad_url_rows)
    bad_url_set = set(bad_url_rows)

    # 5. billing_zip: all rows — no pre-selection rng

    # 6. phone_number: 100 with letters
    bad_phone_rows = rng.sample(range(NROWS), 100)
    bad_phone_set = set(bad_phone_rows)

    # 7. ship_date: 30 before order_date
    ship_before_order_rows = set(rng.sample(range(NROWS), 30))

    # 8. quantity: 5 negative
    neg_qty_rows = rng.sample(range(NROWS), 5)
    neg_qty_set = set(neg_qty_rows)

    # 9. discount_pct: 8 > 100
    bad_discount_rows = rng.sample(range(NROWS), 8)
    bad_discount_set = set(bad_discount_rows)

    # 10. customer_name: 12 numeric
    numeric_name_rows = rng.sample(range(NROWS), 12)
    numeric_name_sorted = sorted(numeric_name_rows)
    numeric_name_set = set(numeric_name_rows)

    # 11. sku: 25 duplicates
    sku_dup_targets = rng.sample(range(NROWS), 25)
    sku_dup_sources = rng.sample(
        [r for r in range(NROWS) if r not in set(sku_dup_targets)], 25
    )

    # 12. rating: 8 values of 6
    bad_rating_rows = rng.sample(range(NROWS), 8)
    bad_rating_set = set(bad_rating_rows)

    # 13/14/15. address_line1/city/state: 40 correlated nulls
    null_address_rows = set(rng.sample(range(NROWS), 40))

    # ------------------------------------------------------------------ #
    # Replay column generation (same order as tier2.py)                  #
    # ------------------------------------------------------------------ #

    from dqbench.generator.tier2 import (
        BASE_CATEGORIES,
        DRIFT_CATEGORIES,
        BAD_URL_EMAILS,
        BAD_EMAIL_VALUES,
        NUMERIC_NAME_VALUES,
        STREET_SUFFIXES,
        ORDER_NOTES_POOL,
        TAGS_POOL,
        JSON_METADATA_POOL,
        INTL_PHONES,
        USER_AGENTS,
        MESSY_NOTES,
        PRODUCT_DESCRIPTIONS,
        CURRENCY_CODES,
        _normal_sample,
    )

    drift_start = 40_000
    ORDER_MEAN = 255.0
    ORDER_STD = 72.0

    start_date = date(2022, 1, 1)
    end_date = date(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    # --- order_date ---
    order_dates_raw: list[date] = [
        start_date + timedelta(days=rng.randint(0, date_range_days))
        for _ in range(NROWS)
    ]

    # --- order_total (detection-only: outlier) — keep messy ---
    order_totals: list[float] = []
    for i in range(NROWS):
        if i in outlier_total_set:
            val = ORDER_MEAN + 3.1 * ORDER_STD + rng.uniform(0, 10)
        else:
            val = max(10.0, _normal_sample(rng, ORDER_MEAN, ORDER_STD))
        order_totals.append(round(val, 2))

    # --- customer_email (detection-only: garbage values) — keep messy ---
    customer_emails: list[str] = []
    for i in range(NROWS):
        if i in bad_email_set:
            customer_emails.append(rng.choice(BAD_EMAIL_VALUES))
        else:
            fn = rng.choice(FIRST_NAMES).lower()
            ln = rng.choice(LAST_NAMES).lower()
            domain = rng.choice(DOMAINS)
            customer_emails.append(f"{fn}.{ln}@{domain}")

    # --- product_category (detection-only: drift) — keep messy ---
    product_categories: list[str] = []
    for i in range(NROWS):
        if i >= drift_start:
            product_categories.append(rng.choice(DRIFT_CATEGORIES))
        else:
            product_categories.append(rng.choice(BASE_CATEGORIES))

    # --- website_url (detection-only: garbage email values) — keep messy ---
    website_urls: list[str] = []
    for i in range(NROWS):
        if i in bad_url_set:
            idx = bad_url_sorted.index(i)
            website_urls.append(BAD_URL_EMAILS[idx])
        else:
            page = rng.choice(["product", "category", "cart", "checkout", "account"])
            pid = rng.randint(1000, 9999)
            website_urls.append(f"https://example.com/{page}/{pid}")

    # --- billing_zip — TRANSFORMABLE: zero-pad to 5 digits ---
    billing_zips_clean: list[str] = []
    for _ in range(NROWS):
        raw_int = rng.randint(0, 99999)
        billing_zips_clean.append(f"{raw_int:05d}")

    # --- phone_number (detection-only: letters in area code — can't recover) ---
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    phone_numbers: list[str] = []
    for i in range(NROWS):
        if i in bad_phone_set:
            d1 = str(rng.randint(2, 9))
            d2 = rng.choice(letters)
            d3 = str(rng.randint(0, 9))
            area_chars = [d1, d2, d3]
            rng.shuffle(area_chars)
            prefix = rng.randint(200, 999)
            line = rng.randint(1000, 9999)
            phone_numbers.append(f"({''.join(area_chars)}) {prefix}-{line}")
        else:
            area = rng.randint(200, 999)
            prefix = rng.randint(200, 999)
            line = rng.randint(1000, 9999)
            phone_numbers.append(f"({area}) {prefix}-{line}")

    # --- ship_date (detection-only: logic_violation) — keep messy ---
    ship_dates: list[str] = []
    for i in range(NROWS):
        od = order_dates_raw[i]
        if i in ship_before_order_rows:
            sd = od - timedelta(days=rng.randint(1, 10))
        else:
            sd = od + timedelta(days=rng.randint(1, 14))
        ship_dates.append(sd.strftime("%Y-%m-%d"))

    # --- quantity (detection-only: invalid_values) — keep messy ---
    quantities: list[int] = []
    for i in range(NROWS):
        if i in neg_qty_set:
            quantities.append(rng.randint(-10, -1))
        else:
            quantities.append(rng.randint(1, 20))

    # --- discount_pct (detection-only: out_of_range) — keep messy ---
    discount_pcts: list[float] = []
    for i in range(NROWS):
        if i in bad_discount_set:
            discount_pcts.append(round(rng.uniform(101.0, 150.0), 2))
        else:
            discount_pcts.append(round(rng.uniform(0.0, 50.0), 2))

    # --- customer_name (detection-only: numeric chars) — keep messy ---
    customer_names: list[str] = []
    for i in range(NROWS):
        if i in numeric_name_set:
            idx = numeric_name_sorted.index(i)
            customer_names.append(NUMERIC_NAME_VALUES[idx])
        else:
            fn = rng.choice(FIRST_NAMES)
            ln = rng.choice(LAST_NAMES)
            customer_names.append(f"{fn} {ln}")

    # --- sku (detection-only: duplicate_values) — keep messy ---
    sku_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    base_skus: list[str] = []
    seen_skus: set[str] = set()
    for _ in range(NROWS):
        while True:
            sku = "SKU-" + "".join(rng.choice(sku_chars) for _ in range(8))
            if sku not in seen_skus:
                seen_skus.add(sku)
                base_skus.append(sku)
                break
    for target, source in zip(sku_dup_targets, sku_dup_sources):
        base_skus[target] = base_skus[source]

    # --- rating (detection-only: enum_violation) — keep messy ---
    ratings: list[int] = [6 if i in bad_rating_set else rng.randint(1, 5) for i in range(NROWS)]

    # --- address_line1 / city / state (detection-only: null_values) — keep messy ---
    address_line1s: list[str | None] = []
    cities: list[str | None] = []
    states: list[str | None] = []
    for i in range(NROWS):
        if i in null_address_rows:
            address_line1s.append(None)
            cities.append(None)
            states.append(None)
        else:
            num = rng.randint(1, 9999)
            suffix = rng.choice(STREET_SUFFIXES)
            address_line1s.append(f"{num} {suffix}")
            city_name, state_code, _ = rng.choice(CITIES)
            cities.append(city_name)
            states.append(state_code)

    # ------------------------------------------------------------------ #
    # Clean columns (same as messy — replay rng in lockstep)             #
    # ------------------------------------------------------------------ #

    order_dates: list[str] = [d.strftime("%Y-%m-%d") for d in order_dates_raw]
    free_text_notes_col: list[str] = [rng.choice(MESSY_NOTES) for _ in range(NROWS)]
    product_descriptions: list[str] = [rng.choice(PRODUCT_DESCRIPTIONS) for _ in range(NROWS)]
    currency_codes: list[str] = [rng.choice(CURRENCY_CODES) for _ in range(NROWS)]
    user_agents: list[str] = [rng.choice(USER_AGENTS) for _ in range(NROWS)]

    ip_addresses: list[str] = []
    for _ in range(NROWS):
        if rng.random() < 0.7:
            ip_addresses.append(
                f"{rng.randint(1, 254)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
            )
        else:
            groups = [format(rng.randint(0, 0xFFFF), "x") for _ in range(8)]
            ip_addresses.append(":".join(groups))

    tags_col: list[str] = [rng.choice(TAGS_POOL) for _ in range(NROWS)]
    json_metadata_col: list[str] = [rng.choice(JSON_METADATA_POOL) for _ in range(NROWS)]
    phone_intl_col: list[str] = [rng.choice(INTL_PHONES) for _ in range(NROWS)]

    address_line2s: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.60:
            address_line2s.append(None)
        else:
            address_line2s.append(rng.choice(["Apt 4B", "Suite 100", "Unit 7", "Floor 3", "Apt 12A", "Suite 200"]))

    order_notes_col: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.80:
            order_notes_col.append(None)
        else:
            order_notes_col.append(rng.choice(ORDER_NOTES_POOL))

    code_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    referral_codes: list[str] = [
        "REF-" + "".join(rng.choice(code_chars) for _ in range(6))
        for _ in range(NROWS)
    ]

    session_ids: list[str] = []
    for _ in range(NROWS):
        p1 = format(rng.randint(0, 0xFFFFFFFF), "08x")
        p2 = format(rng.randint(0, 0xFFFF), "04x")
        p3 = format((rng.randint(0, 0xFFFF) & 0x0FFF) | 0x4000, "04x")
        p4 = format((rng.randint(0, 0xFFFF) & 0x3FFF) | 0x8000, "04x")
        p5 = format(rng.randint(0, 0xFFFFFFFFFFFF), "012x")
        session_ids.append(f"{p1}-{p2}-{p3}-{p4}-{p5}")

    from datetime import datetime, timezone
    created_ats: list[str] = []
    for i in range(NROWS):
        od = order_dates_raw[i]
        dt = datetime(od.year, od.month, od.day,
                      rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59),
                      tzinfo=timezone.utc)
        created_ats.append(dt.isoformat())

    updated_ats: list[str] = []
    for i in range(NROWS):
        od = order_dates_raw[i]
        extra_days = rng.randint(0, 30)
        dt = datetime(od.year, od.month, od.day,
                      rng.randint(0, 23), rng.randint(0, 59), 0,
                      tzinfo=timezone.utc) + timedelta(days=extra_days)
        updated_ats.append(dt.isoformat())

    df = pl.DataFrame(
        {
            # Planted (15)
            "order_total":      order_totals,
            "customer_email":   customer_emails,
            "product_category": product_categories,
            "website_url":      website_urls,
            "billing_zip":      billing_zips_clean,   # TRANSFORMED: zero-padded strings
            "phone_number":     phone_numbers,
            "ship_date":        ship_dates,
            "quantity":         quantities,
            "discount_pct":     discount_pcts,
            "customer_name":    customer_names,
            "sku":              base_skus,
            "rating":           ratings,
            "address_line1":    address_line1s,
            "city":             cities,
            "state":            states,
            # Clean (15)
            "order_date":          order_dates,
            "free_text_notes":     free_text_notes_col,
            "product_description": product_descriptions,
            "currency_code":       currency_codes,
            "user_agent":          user_agents,
            "ip_address":          ip_addresses,
            "tags":                tags_col,
            "json_metadata":       json_metadata_col,
            "phone_intl":          phone_intl_col,
            "address_line2":       address_line2s,
            "order_notes":         order_notes_col,
            "referral_code":       referral_codes,
            "session_id":          session_ids,
            "created_at":          created_ats,
            "updated_at":          updated_ats,
        }
    )
    return df


def generate_clean_tier3() -> pl.DataFrame:
    """Replay the tier3 RNG to produce a clean (ground-truth) DataFrame.

    Transformable columns are normalised to their canonical form.
    Detection-only columns keep the messy value (they won't be scored).

    Transformable issues in tier 3:
    - provider_name: zero-width Unicode chars embedded → strip them
    - diagnosis_desc: smart/curly quotes → replace with straight ASCII quotes

    Detection-only (kept as messy):
    - npi_number: luhn failure (can't recover original valid NPI)
    - patient_state: cross_column (state/zip mismatch — ambiguous which is wrong)
    - service_date: logic_violation (before date_of_birth)
    - claim_notes: encoding (Latin-1 strings are replacement values — no original)
    - claim_amount: logic_violation (exceeds policy max)
    - service_day: semantic (weekend for weekday-only providers)
    - submit_date: logic_violation (before service_date)
    - record_number: sequence_gap
    - patient_age: logic_violation (doesn't match dob)
    - procedure_code: invalid_format (garbage values)
    - insurance_id: logic_violation (wrong prefix)
    - patient_zip: invalid_values (nonexistent zips)
    - dosage_amount: outlier_values
    - lab_result: outlier_values (bimodal gap)
    - admission_date/discharge_date: logic_violation (admission after discharge)
    - primary_dx: ICD-9 mixed in (invalid_format — detection only)
    - secondary_dx: conflicting codes (detection only)
    - charge_code: duplicate_values
    - patient_name: wrong_type (numeric chars — no correction)
    - referring_npi: luhn failure
    - auth_number: wrong length
    - payment_amount: invalid_values (negatives)
    """
    rng = random.Random(42)
    NROWS = _T3_NROWS

    from dqbench.generator.tier3 import (
        PROVIDER_TYPES,
        INSURANCE_PREFIXES,
        WRONG_PREFIXES,
        VALID_ICD10_CODES,
        ICD9_CODES,
        INVALID_PROCEDURE_CODES,
        FACILITY_CODES,
        BILLING_CODES,
        MODIFIER_CODES,
        PLACE_OF_SERVICE,
        REVENUE_CODES,
        DRG_CODES,
        REFERRAL_SOURCES,
        SERVICE_TYPES,
        CLAIM_STATUSES,
        GENDERS,
        ATTENDING_PHYSICIANS,
        PRIMARY_INSURANCES,
        SECONDARY_INSURANCES,
        CONFLICTING_DX_PAIRS,
        LATIN1_STRINGS,
        ZERO_WIDTH_CHARS,
        STRAIGHT_QUOTE_PHRASES,
        PRIOR_AUTH_FLAGS,
        NUMERIC_PATIENT_NAMES,
        NONEXISTENT_ZIPS,
        STATE_ABBREVS,
        ZIP_STATE_MAP,
        ZIP_STATE_WRONG,
        WEEKDAY_ONLY_PROVIDERS,
        _normal_sample,
        _rand_date,
        _next_weekday,
        _to_weekend,
        _make_valid_npi,
        _corrupt_npi,
    )

    # ------------------------------------------------------------------ #
    # Pre-select issue rows (same order as tier3.py)                      #
    # ------------------------------------------------------------------ #

    # 1. npi_number: 50 fail Luhn
    bad_npi_rows = rng.sample(range(NROWS), 50)
    bad_npi_set = set(bad_npi_rows)

    # 2. patient_state: 30 wrong for zip prefix
    wrong_state_rows = rng.sample(range(NROWS), 30)
    wrong_state_set = set(wrong_state_rows)

    # 3. service_date: 20 before date_of_birth
    service_before_dob_rows = rng.sample(range(NROWS), 20)
    service_before_dob_set = set(service_before_dob_rows)

    # 4. claim_notes: 15 Latin-1 chars
    latin1_note_rows = rng.sample(range(NROWS), 15)
    latin1_note_set = set(latin1_note_rows)

    # 5. provider_name: 10 zero-width Unicode chars
    zwsp_provider_rows = rng.sample(range(NROWS), 10)
    zwsp_provider_set = set(zwsp_provider_rows)

    # 6. diagnosis_desc: 25 smart quotes
    smart_quote_rows = rng.sample(range(NROWS), 25)
    smart_quote_set = set(smart_quote_rows)

    # 7. claim_amount vs policy_max_amount: 12 exceed policy max
    exceed_policy_rows = rng.sample(range(NROWS), 12)
    exceed_policy_set = set(exceed_policy_rows)

    # 8. service_day: 20 weekend dates for weekday-only providers
    weekend_service_rows = rng.sample(range(NROWS), 20)
    weekend_service_set = set(weekend_service_rows)

    # 9. submit_date: 15 before service_date
    submit_before_service_rows = rng.sample(range(NROWS), 15)
    submit_before_service_set = set(submit_before_service_rows)

    # 10. record_number: 8 gaps
    gap_record_rows = sorted(rng.sample(range(1, NROWS), 8))
    gap_set = set(gap_record_rows)

    # 11. patient_age: 5 don't match date_of_birth
    wrong_age_rows = rng.sample(range(NROWS), 5)
    wrong_age_set = set(wrong_age_rows)

    # 12. procedure_code: 18 invalid ICD-10 format
    bad_procedure_rows = rng.sample(range(NROWS), 18)
    bad_procedure_set = set(bad_procedure_rows)

    # 13. insurance_id: 22 wrong prefix for provider_type
    bad_insurance_rows = rng.sample(range(NROWS), 22)
    bad_insurance_set = set(bad_insurance_rows)

    # 14. patient_zip: 10 nonexistent zips
    nonexistent_zip_rows = rng.sample(range(NROWS), 10)
    nonexistent_zip_set = set(nonexistent_zip_rows)

    # 15. dosage_amount: 8 extreme outliers
    dosage_outlier_rows = rng.sample(range(NROWS), 8)
    dosage_outlier_set = set(dosage_outlier_rows)

    # 16. lab_result: 6 values in bimodal gap
    lab_gap_rows = rng.sample(range(NROWS), 6)
    lab_gap_set = set(lab_gap_rows)

    # 17/18. admission_date / discharge_date: 12 correlated
    admission_after_discharge_rows = set(rng.sample(range(NROWS), 12))

    # 19. primary_dx: 15 ICD-9 codes mixed in
    icd9_mix_rows = rng.sample(range(NROWS), 15)
    icd9_mix_set = set(icd9_mix_rows)

    # 20. secondary_dx: 10 conflict with primary_dx
    conflicting_dx_rows = rng.sample(
        [r for r in range(NROWS) if r not in icd9_mix_set], 10
    )
    conflicting_dx_set = set(conflicting_dx_rows)

    # 21. charge_code: 20 duplicates
    charge_dup_targets = rng.sample(range(NROWS), 20)
    charge_dup_sources = rng.sample(
        [r for r in range(NROWS) if r not in set(charge_dup_targets)], 20
    )

    # 22. patient_name: 8 numeric chars
    numeric_patient_rows = rng.sample(range(NROWS), 8)
    numeric_patient_set = set(numeric_patient_rows)
    numeric_patient_sorted = sorted(numeric_patient_rows)

    # 23. referring_npi: 25 fail Luhn
    bad_ref_npi_rows = rng.sample(range(NROWS), 25)
    bad_ref_npi_set = set(bad_ref_npi_rows)

    # 24. auth_number: 5 wrong length
    bad_auth_rows = rng.sample(range(NROWS), 5)
    bad_auth_set = set(bad_auth_rows)

    # 25. payment_amount: 7 negative
    neg_payment_rows = rng.sample(range(NROWS), 7)
    neg_payment_set = set(neg_payment_rows)

    # ------------------------------------------------------------------ #
    # Generate base clean data (same as tier3.py)                        #
    # ------------------------------------------------------------------ #

    dob_start = date(1930, 1, 1)
    dob_end = date(2005, 12, 31)
    service_start = date(2020, 1, 1)
    service_end = date(2024, 12, 31)

    dobs: list[date] = [_rand_date(rng, dob_start, dob_end) for _ in range(NROWS)]

    patient_zips: list[str] = []
    for i in range(NROWS):
        if i in nonexistent_zip_set:
            patient_zips.append(rng.choice(NONEXISTENT_ZIPS))
        else:
            patient_zips.append(f"{rng.randint(10000, 89999):05d}")

    patient_states: list[str] = []
    for i in range(NROWS):
        z = patient_zips[i]
        prefix = z[0]
        if i in wrong_state_set:
            wrong_opts = ZIP_STATE_WRONG.get(prefix, STATE_ABBREVS)
            patient_states.append(rng.choice(wrong_opts))
        else:
            correct_state = ZIP_STATE_MAP.get(prefix, rng.choice(STATE_ABBREVS))
            patient_states.append(correct_state)

    provider_types: list[str] = [rng.choice(PROVIDER_TYPES) for _ in range(NROWS)]

    service_dates_raw: list[date] = []
    for i in range(NROWS):
        if i in service_before_dob_set:
            dob = dobs[i]
            service_dates_raw.append(dob - timedelta(days=rng.randint(1, 3650)))
        else:
            service_dates_raw.append(_rand_date(rng, service_start, service_end))

    service_days_raw: list[date] = []
    for i in range(NROWS):
        sd = service_dates_raw[i]
        if i in weekend_service_set and provider_types[i] in WEEKDAY_ONLY_PROVIDERS:
            service_days_raw.append(_to_weekend(sd, rng))
        else:
            if provider_types[i] in WEEKDAY_ONLY_PROVIDERS:
                service_days_raw.append(_next_weekday(sd))
            else:
                service_days_raw.append(sd)

    service_days: list[str] = [d.strftime("%Y-%m-%d") for d in service_days_raw]
    service_dates: list[str] = [d.strftime("%Y-%m-%d") for d in service_dates_raw]

    submit_dates: list[str] = []
    for i in range(NROWS):
        svc = service_dates_raw[i]
        if i in submit_before_service_set:
            sub = svc - timedelta(days=rng.randint(1, 30))
        else:
            sub = svc + timedelta(days=rng.randint(0, 45))
        submit_dates.append(sub.strftime("%Y-%m-%d"))

    admission_dates: list[str] = []
    discharge_dates: list[str] = []
    for i in range(NROWS):
        svc = service_dates_raw[i]
        if i in admission_after_discharge_rows:
            disc = svc + timedelta(days=rng.randint(1, 5))
            adm = disc + timedelta(days=rng.randint(1, 3))
        else:
            adm = svc
            disc = svc + timedelta(days=rng.randint(0, 14))
        admission_dates.append(adm.strftime("%Y-%m-%d"))
        discharge_dates.append(disc.strftime("%Y-%m-%d"))

    reference_date = date(2024, 1, 1)
    patient_ages: list[int] = []
    for i in range(NROWS):
        dob = dobs[i]
        correct_age = (reference_date - dob).days // 365
        if i in wrong_age_set:
            wrong_age = correct_age + rng.choice([-10, -8, -7, 10, 12, 15])
            patient_ages.append(max(1, wrong_age))
        else:
            patient_ages.append(correct_age)

    # npi_number (detection-only: Luhn fail — keep messy)
    npi_numbers: list[str] = []
    for i in range(NROWS):
        valid_npi = _make_valid_npi(rng)
        if i in bad_npi_set:
            npi_numbers.append(_corrupt_npi(valid_npi, rng))
        else:
            npi_numbers.append(valid_npi)

    # referring_npi (detection-only: Luhn fail — keep messy)
    referring_npis: list[str] = []
    for i in range(NROWS):
        valid_npi = _make_valid_npi(rng)
        if i in bad_ref_npi_set:
            referring_npis.append(_corrupt_npi(valid_npi, rng))
        else:
            referring_npis.append(valid_npi)

    # claim_notes (detection-only: Latin-1 replacement values — keep messy)
    clean_claim_notes = [
        "Routine follow-up visit.",
        "Patient reports improvement.",
        "Labs ordered for monitoring.",
        "Prescription refilled.",
        "Referral to specialist placed.",
        "No acute distress observed.",
        "Patient education provided.",
        "Follow-up scheduled in 4 weeks.",
        "Imaging ordered.",
        "Vitals within normal range.",
    ]
    claim_notes: list[str] = []
    for i in range(NROWS):
        if i in latin1_note_set:
            claim_notes.append(rng.choice(LATIN1_STRINGS))
        else:
            claim_notes.append(rng.choice(clean_claim_notes))

    # provider_name — TRANSFORMABLE: strip zero-width chars
    base_provider_names = [
        "Dr. John Smith", "Dr. Mary Johnson", "Dr. Robert Brown",
        "Northwest Medical Center", "Eastside Clinic",
        "Dr. Jennifer Garcia", "Memorial Hospital",
        "Dr. Michael Miller", "City Health Lab",
        "Dr. Patricia Davis",
    ]
    provider_names_clean: list[str] = []
    _zwc_set = set(ZERO_WIDTH_CHARS)
    for i in range(NROWS):
        name = rng.choice(base_provider_names)
        if i in zwsp_provider_set:
            zwc = rng.choice(ZERO_WIDTH_CHARS)
            mid = len(name) // 2
            name = name[:mid] + zwc + name[mid:]
        # Clean: strip all zero-width characters
        provider_names_clean.append("".join(c for c in name if c not in _zwc_set))

    # diagnosis_desc — TRANSFORMABLE: smart quotes → straight ASCII quotes
    smart_quote_row_idx: dict[int, int] = {
        r: j for j, r in enumerate(sorted(smart_quote_rows))
    }
    # SMART_QUOTE_PHRASES[i] corresponds to STRAIGHT_QUOTE_PHRASES[i]
    diagnosis_descs_clean: list[str] = []
    for i in range(NROWS):
        if i in smart_quote_set:
            idx = smart_quote_row_idx[i]
            # Use the straight-quote version of the same phrase
            diagnosis_descs_clean.append(STRAIGHT_QUOTE_PHRASES[idx % len(STRAIGHT_QUOTE_PHRASES)])
        else:
            diagnosis_descs_clean.append(rng.choice(STRAIGHT_QUOTE_PHRASES))

    # policy_max_amount (clean)
    policy_max_amounts: list[float] = []
    for _ in range(NROWS):
        policy_max_amounts.append(round(rng.choice([5000.0, 10000.0, 25000.0, 50000.0, 100000.0]), 2))

    # claim_amount (detection-only: logic_violation) — keep messy
    claim_amounts: list[float] = []
    for i in range(NROWS):
        if i in exceed_policy_set:
            factor = 1.1 + rng.random() * 0.4
            claim_amounts.append(round(policy_max_amounts[i] * factor, 2))
        else:
            claim_amounts.append(round(rng.uniform(50.0, policy_max_amounts[i] * 0.95), 2))

    # record_number (detection-only: sequence_gap) — keep messy
    record_numbers: list[int] = []
    seq = 1
    for i in range(NROWS):
        if i in gap_set:
            seq += 2
        record_numbers.append(seq)
        seq += 1

    # procedure_code (detection-only: invalid format) — keep messy
    procedure_codes: list[str] = []
    for i in range(NROWS):
        if i in bad_procedure_set:
            procedure_codes.append(rng.choice(INVALID_PROCEDURE_CODES))
        else:
            procedure_codes.append(rng.choice(VALID_ICD10_CODES))

    # insurance_id (detection-only: wrong prefix) — keep messy
    insurance_ids: list[str] = []
    for i in range(NROWS):
        ptype = provider_types[i]
        if i in bad_insurance_set:
            wrong_prefix = rng.choice(WRONG_PREFIXES[ptype])
            suffix = format(rng.randint(100000, 999999), "06d")
            insurance_ids.append(f"{wrong_prefix}{suffix}")
        else:
            correct_prefix = INSURANCE_PREFIXES[ptype]
            suffix = format(rng.randint(100000, 999999), "06d")
            insurance_ids.append(f"{correct_prefix}{suffix}")

    # dosage_amount (detection-only: outlier) — keep messy
    DOSAGE_MEAN = 250.0
    DOSAGE_STD = 50.0
    dosage_amounts: list[float] = []
    for i in range(NROWS):
        if i in dosage_outlier_set:
            val = DOSAGE_MEAN + (4.0 + rng.random() * 2.0) * DOSAGE_STD
        else:
            val = max(1.0, _normal_sample(rng, DOSAGE_MEAN, DOSAGE_STD))
        dosage_amounts.append(round(val, 2))

    # lab_result (detection-only: bimodal gap) — keep messy
    lab_results: list[float] = []
    for i in range(NROWS):
        if i in lab_gap_set:
            lab_results.append(round(rng.uniform(42.0, 58.0), 2))
        else:
            if rng.random() < 0.5:
                val = _normal_sample(rng, 20.0, 5.0)
                lab_results.append(round(min(38.0, max(0.0, val)), 2))
            else:
                val = _normal_sample(rng, 80.0, 8.0)
                lab_results.append(round(max(62.0, min(120.0, val)), 2))

    # primary_dx (detection-only: ICD-9 mixed in) — keep messy
    primary_dxs: list[str] = []
    for i in range(NROWS):
        if i in icd9_mix_set:
            primary_dxs.append(rng.choice(ICD9_CODES))
        else:
            primary_dxs.append(rng.choice(VALID_ICD10_CODES))

    # secondary_dx (detection-only: conflicting codes) — keep messy
    secondary_dxs: list[str | None] = []
    for i in range(NROWS):
        if i in conflicting_dx_set:
            pdx = primary_dxs[i]
            pair = rng.choice(CONFLICTING_DX_PAIRS)
            if pair[0] == pdx:
                secondary_dxs.append(pair[1])
            else:
                secondary_dxs.append(pair[0])
        else:
            secondary_dxs.append(rng.choice(VALID_ICD10_CODES + [None, None]))  # type: ignore[arg-type]

    # charge_code (detection-only: duplicate_values) — keep messy
    charge_codes_base: list[str] = []
    seen_cc: set[str] = set()
    cc_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for _ in range(NROWS):
        while True:
            cc = "CHG-" + "".join(rng.choice(cc_chars) for _ in range(6))
            if cc not in seen_cc:
                seen_cc.add(cc)
                charge_codes_base.append(cc)
                break
    for target, source in zip(charge_dup_targets, charge_dup_sources):
        charge_codes_base[target] = charge_codes_base[source]

    # patient_name (detection-only: numeric chars) — keep messy
    patient_names: list[str] = []
    for i in range(NROWS):
        if i in numeric_patient_set:
            idx = numeric_patient_sorted.index(i)
            patient_names.append(NUMERIC_PATIENT_NAMES[idx % len(NUMERIC_PATIENT_NAMES)])
        else:
            fn = rng.choice(FIRST_NAMES)
            ln = rng.choice(LAST_NAMES)
            patient_names.append(f"{fn} {ln}")

    # auth_number (detection-only: wrong length) — keep messy
    auth_numbers: list[str] = []
    for i in range(NROWS):
        if i in bad_auth_set:
            bad_len = rng.choice([6, 15])
            auth_numbers.append("".join(str(rng.randint(0, 9)) for _ in range(bad_len)))
        else:
            auth_numbers.append("".join(str(rng.randint(0, 9)) for _ in range(10)))

    # payment_amount (detection-only: negative values) — keep messy
    payment_amounts: list[float] = []
    for i in range(NROWS):
        if i in neg_payment_set:
            payment_amounts.append(round(-rng.uniform(10.0, 500.0), 2))
        else:
            payment_amounts.append(round(rng.uniform(0.0, claim_amounts[i]), 2))

    # ------------------------------------------------------------------ #
    # Clean columns (replay rng in lockstep)                             #
    # ------------------------------------------------------------------ #

    patient_ids: list[str] = [f"PAT-{i+1:07d}" for i in range(NROWS)]
    genders: list[str] = [rng.choice(GENDERS) for _ in range(NROWS)]
    facility_codes_col: list[str] = [rng.choice(FACILITY_CODES) for _ in range(NROWS)]
    billing_codes_col: list[str] = [rng.choice(BILLING_CODES) for _ in range(NROWS)]

    modifier_codes: list[str | None] = []
    for _ in range(NROWS):
        modifier_codes.append(None if rng.random() < 0.4 else rng.choice(MODIFIER_CODES))

    pos_col: list[str] = [rng.choice(PLACE_OF_SERVICE) for _ in range(NROWS)]
    revenue_codes_col: list[str] = [rng.choice(REVENUE_CODES) for _ in range(NROWS)]

    drg_codes: list[str | None] = []
    for _ in range(NROWS):
        drg_codes.append(None if rng.random() < 0.5 else rng.choice(DRG_CODES))

    attending_physicians: list[str] = [rng.choice(ATTENDING_PHYSICIANS) for _ in range(NROWS)]
    referral_sources_col: list[str] = [rng.choice(REFERRAL_SOURCES) for _ in range(NROWS)]

    patient_phones: list[str] = []
    for _ in range(NROWS):
        area = rng.randint(200, 999)
        prefix = rng.randint(200, 999)
        line = rng.randint(1000, 9999)
        patient_phones.append(f"({area}) {prefix}-{line}")

    patient_emails: list[str] = []
    for _ in range(NROWS):
        fn = rng.choice(FIRST_NAMES).lower()
        ln = rng.choice(LAST_NAMES).lower()
        domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "healthmail.org"])
        patient_emails.append(f"{fn}.{ln}@{domain}")

    emergency_contacts: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.3:
            emergency_contacts.append(None)
        else:
            fn = rng.choice(FIRST_NAMES)
            ln = rng.choice(LAST_NAMES)
            emergency_contacts.append(f"{fn} {ln}")

    primary_insurances: list[str] = [rng.choice(PRIMARY_INSURANCES) for _ in range(NROWS)]

    secondary_insurances_col: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.7:
            secondary_insurances_col.append(None)
        else:
            secondary_insurances_col.append(rng.choice([s for s in SECONDARY_INSURANCES if s is not None]))

    copay_amounts: list[float] = [
        round(rng.choice([0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0]), 2)
        for _ in range(NROWS)
    ]
    deductible_met: list[str] = [rng.choice(["Y", "N"]) for _ in range(NROWS)]
    prior_auth_flags_col: list[str] = [rng.choice(PRIOR_AUTH_FLAGS) for _ in range(NROWS)]
    service_types: list[str] = [rng.choice(SERVICE_TYPES) for _ in range(NROWS)]
    claim_statuses: list[str] = [rng.choice(CLAIM_STATUSES) for _ in range(NROWS)]

    adjudication_dates: list[str | None] = []
    for i in range(NROWS):
        if rng.random() < 0.2:
            adjudication_dates.append(None)
        else:
            sub_date = date.fromisoformat(submit_dates[i])
            adj_date = sub_date + timedelta(days=rng.randint(1, 60))
            adjudication_dates.append(adj_date.strftime("%Y-%m-%d"))

    remittance_amounts: list[float] = []
    for i in range(NROWS):
        pa = payment_amounts[i]
        if pa < 0:
            remittance_amounts.append(0.0)
        else:
            remittance_amounts.append(round(pa * rng.uniform(0.95, 1.0), 2))

    dobs_str: list[str] = [d.strftime("%Y-%m-%d") for d in dobs]

    secondary_dxs_clean_col: list[str | None] = [
        v if isinstance(v, str) else None for v in secondary_dxs
    ]

    df = pl.DataFrame(
        {
            # Planted (25)
            "npi_number":       npi_numbers,
            "patient_state":    patient_states,
            "service_date":     service_dates,
            "claim_notes":      claim_notes,
            "provider_name":    provider_names_clean,    # TRANSFORMED: zero-width chars stripped
            "diagnosis_desc":   diagnosis_descs_clean,   # TRANSFORMED: smart quotes → straight
            "claim_amount":     claim_amounts,
            "service_day":      service_days,
            "submit_date":      submit_dates,
            "record_number":    record_numbers,
            "patient_age":      patient_ages,
            "procedure_code":   procedure_codes,
            "insurance_id":     insurance_ids,
            "patient_zip":      patient_zips,
            "dosage_amount":    dosage_amounts,
            "lab_result":       lab_results,
            "admission_date":   admission_dates,
            "discharge_date":   discharge_dates,
            "primary_dx":       primary_dxs,
            "secondary_dx":     secondary_dxs_clean_col,
            "charge_code":      charge_codes_base,
            "patient_name":     patient_names,
            "referring_npi":    referring_npis,
            "auth_number":      auth_numbers,
            "payment_amount":   payment_amounts,
            # Clean (25)
            "patient_id":           patient_ids,
            "date_of_birth":        dobs_str,
            "gender":               genders,
            "policy_max_amount":    policy_max_amounts,
            "provider_type":        provider_types,
            "facility_code":        facility_codes_col,
            "billing_code":         billing_codes_col,
            "modifier_code":        modifier_codes,
            "place_of_service":     pos_col,
            "revenue_code":         revenue_codes_col,
            "drg_code":             drg_codes,
            "attending_physician":  attending_physicians,
            "referral_source":      referral_sources_col,
            "patient_phone":        patient_phones,
            "patient_email":        patient_emails,
            "emergency_contact":    emergency_contacts,
            "primary_insurance":    primary_insurances,
            "secondary_insurance":  secondary_insurances_col,
            "copay_amount":         copay_amounts,
            "deductible_met":       deductible_met,
            "prior_auth_flag":      prior_auth_flags_col,
            "service_type":         service_types,
            "claim_status":         claim_statuses,
            "adjudication_date":    adjudication_dates,
            "remittance_amount":    remittance_amounts,
        }
    )
    return df


def generate_clean_csvs(cache_dir: Path) -> None:
    """Generate clean ground truth CSVs for all three tiers."""
    # Tier 1 — full clean generation
    clean1 = generate_clean_tier1()
    clean1.write_csv(cache_dir / "tier1" / "data_clean.csv")

    # Tier 2 — full clean generation
    clean2 = generate_clean_tier2()
    clean2.write_csv(cache_dir / "tier2" / "data_clean.csv")

    # Tier 3 — full clean generation
    clean3 = generate_clean_tier3()
    clean3.write_csv(cache_dir / "tier3" / "data_clean.csv")
