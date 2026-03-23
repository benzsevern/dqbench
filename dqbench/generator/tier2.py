"""Tier 2 dataset generator — 50,000-row e-commerce orders with planted issues."""
from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone

import polars as pl

from dqbench.ground_truth import GroundTruth, PlantedColumn
from dqbench.generator.utils import (
    FIRST_NAMES,
    LAST_NAMES,
    DOMAINS,
    CITIES,
)

NROWS = 50_000

# Product categories — 10 base + 3 drift
BASE_CATEGORIES = [
    "Electronics", "Clothing", "Books", "Home & Garden", "Sports",
    "Toys", "Beauty", "Automotive", "Food & Grocery", "Office Supplies",
]
DRIFT_CATEGORIES = ["Cryptocurrency", "NFT Art", "Metaverse Goods"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
]

TAGS_POOL = [
    "sale,clearance", "new-arrival", "bestseller,popular", "limited-edition",
    "bundle,discount", "seasonal", "flash-sale,24hr", "exclusive",
    "eco-friendly,sustainable", "gift-wrap,holiday", "refurbished", "pre-order",
]

INTL_PHONES = [
    "+1-555-123-4567", "+1-800-555-0199", "+44 20 7946 0958", "+44 7911 123456",
    "+49 30 12345678", "+33 1 23 45 67 89", "+61 2 1234 5678", "+81 3-1234-5678",
    "+55 11 91234-5678", "+52 55 1234 5678",
]

CURRENCY_CODES = ["USD", "EUR", "GBP"]

MESSY_NOTES = [
    "See order #12345 for reference. Visit https://support.example.com",
    "Customer called 3 times. Amount refunded: $45.00",
    "Note: qty=2, discount=15%, total=85.00",
    "Promo code SAVE20 applied. Expires 2024-12-31.",
    "Delivery attempted at 10:30am. Re-attempt scheduled.",
    "Order linked to B2B account #ENT-7890.",
    "Package weight: 2.5kg. Dimensions: 30x20x10cm.",
    "Customer rating: 4/5. Feedback: 'Great product!'",
]

PRODUCT_DESCRIPTIONS = [
    "High-quality product with durable materials &amp; excellent finish.",
    "Compact &amp; lightweight design, perfect for travel. <b>New model</b>.",
    "Best-in-class performance &mdash; see full specs below.",
    "Customer favorite &amp; top-rated. Limited stock available.",
    "Premium grade &mdash; made with eco-friendly materials.",
    "Ergonomic design &amp; versatile functionality.",
    "Pack of 3 &mdash; great value for everyday use.",
    "Compatible with all major brands &amp; models.",
]

JSON_METADATA_POOL = [
    '{"source": "web", "ab_test": "variant_b", "promo": null}',
    '{"source": "mobile", "ab_test": "control", "promo": "SAVE10"}',
    '{"source": "email", "ab_test": "variant_a", "promo": "WELCOME20"}',
    '{"source": "social", "ab_test": "control", "promo": null}',
    '{"source": "web", "ab_test": "variant_c", "promo": "FLASH50"}',
    '{"source": "partner", "ab_test": "control", "promo": "PARTNER15"}',
]

ORDER_NOTES_POOL = [
    "Customer requested expedited shipping.",
    "Gift order — include gift receipt.",
    "Please leave at door.",
    "Call before delivery.",
    "Fragile items — handle with care.",
    "Repeat customer — apply loyalty discount.",
]

STREET_SUFFIXES = [
    "Main St", "Oak Ave", "Elm Rd", "Maple Dr", "Cedar Ln",
    "Pine Blvd", "Park Way", "Lake Dr", "Hill Rd", "River Rd",
    "Sunset Blvd", "Commerce Dr", "Industrial Pkwy", "Market St",
]

STATE_ABBREVS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

BAD_URL_EMAILS = [
    "customer.support@example.com",
    "info@shop.example.com",
    "orders@ecommerce.biz",
    "noreply@store.com",
    "help@marketplace.io",
    "contact@retailer.net",
    "admin@website.org",
    "service@onlineshop.co",
    "support@deals.com",
    "team@orders.net",
    "notify@checkout.com",
    "sales@product.io",
    "feedback@reviews.org",
    "billing@payments.net",
    "delivery@logistics.com",
]

BAD_EMAIL_VALUES = [
    "not-an-email", "user@@domain.com", "plainaddress", "@nodomain.com",
    "user@", "missing@", "two@@at.com", "no-at-sign", "spaces in@email.com",
    "user@domain", "email.without.at", "just-text", "123456", "N/A",
    "@.com", "user@.com", ".user@domain.com", "user.@domain.com",
    "user@-domain.com", "user@domain-.com",
]

NUMERIC_NAME_VALUES = [
    "J0hn Sm1th", "M4ry J4n3", "R0b3rt Br0wn", "P4tr1c14 D4v1s",
    "J0hn W1ll14ms", "J3nn1f3r G4rc14", "M1ch43l M1ll3r", "L1nd4 W1ls0n",
    "W1ll14m M00r3", "B4rb4r4 T4yl0r", "D4v1d J4cks0n", "3l1z4b3th M4rt1n",
]


def _normal_sample(rng: random.Random, mean: float, std: float) -> float:
    """Box-Muller transform for normal distribution without numpy."""
    u1 = rng.random()
    u2 = rng.random()
    while u1 == 0.0:
        u1 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


def generate_tier2() -> tuple[pl.DataFrame, GroundTruth]:
    """Generate the Tier 2 benchmark dataset.

    Returns a 50,000-row e-commerce orders DataFrame with 30 columns
    (15 planted issues, 15 clean) and the corresponding GroundTruth.
    """
    rng = random.Random(42)

    # ------------------------------------------------------------------ #
    # Pre-select which rows get which issues (all indices 0-based)        #
    # ------------------------------------------------------------------ #

    ORDER_MEAN = 255.0
    ORDER_STD = 72.0

    # 1. order_total: 20 outliers
    outlier_total_rows = rng.sample(range(NROWS), 20)

    # 2. customer_email: 200 invalid
    bad_email_rows = rng.sample(range(NROWS), 200)

    # 3. product_category: last 10K rows are drift (rows 40000-49999)
    drift_start = 40_000

    # 4. website_url: 15 email-format values
    bad_url_rows = rng.sample(range(NROWS), 15)
    bad_url_sorted = sorted(bad_url_rows)

    # 5. billing_zip: all rows stored as Int64 (schema-level issue)

    # 6. phone_number: 100 with letters
    bad_phone_rows = rng.sample(range(NROWS), 100)

    # 7. ship_date: 30 before order_date
    ship_before_order_rows = set(rng.sample(range(NROWS), 30))

    # 8. quantity: 5 negative
    neg_qty_rows = rng.sample(range(NROWS), 5)

    # 9. discount_pct: 8 > 100
    bad_discount_rows = rng.sample(range(NROWS), 8)

    # 10. customer_name: 12 numeric
    numeric_name_rows = rng.sample(range(NROWS), 12)
    numeric_name_sorted = sorted(numeric_name_rows)

    # 11. sku: 25 duplicates
    sku_dup_targets = rng.sample(range(NROWS), 25)
    sku_dup_sources = rng.sample(
        [r for r in range(NROWS) if r not in set(sku_dup_targets)], 25
    )

    # 12. rating: 8 values of 6
    bad_rating_rows = rng.sample(range(NROWS), 8)

    # 13/14/15. address_line1/city/state: 40 correlated nulls
    null_address_rows = set(rng.sample(range(NROWS), 40))

    # ------------------------------------------------------------------ #
    # Build columns row by row                                            #
    # ------------------------------------------------------------------ #

    start_date = date(2022, 1, 1)
    end_date = date(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    # --- order_date (clean) ---
    order_dates_raw: list[date] = [
        start_date + timedelta(days=rng.randint(0, date_range_days))
        for _ in range(NROWS)
    ]

    # --- order_total (planted) ---
    outlier_total_set = set(outlier_total_rows)
    order_totals: list[float] = []
    for i in range(NROWS):
        if i in outlier_total_set:
            val = ORDER_MEAN + 3.1 * ORDER_STD + rng.uniform(0, 10)
        else:
            val = max(10.0, _normal_sample(rng, ORDER_MEAN, ORDER_STD))
        order_totals.append(round(val, 2))

    # --- customer_email (planted) ---
    bad_email_set = set(bad_email_rows)
    customer_emails: list[str] = []
    for i in range(NROWS):
        if i in bad_email_set:
            customer_emails.append(rng.choice(BAD_EMAIL_VALUES))
        else:
            fn = rng.choice(FIRST_NAMES).lower()
            ln = rng.choice(LAST_NAMES).lower()
            domain = rng.choice(DOMAINS)
            customer_emails.append(f"{fn}.{ln}@{domain}")

    # --- product_category (planted — drift) ---
    product_categories: list[str] = []
    for i in range(NROWS):
        if i >= drift_start:
            product_categories.append(rng.choice(DRIFT_CATEGORIES))
        else:
            product_categories.append(rng.choice(BASE_CATEGORIES))

    # --- website_url (planted) ---
    bad_url_set = set(bad_url_rows)
    website_urls: list[str] = []
    for i in range(NROWS):
        if i in bad_url_set:
            idx = bad_url_sorted.index(i)
            website_urls.append(BAD_URL_EMAILS[idx])
        else:
            page = rng.choice(["product", "category", "cart", "checkout", "account"])
            pid = rng.randint(1000, 9999)
            website_urls.append(f"https://example.com/{page}/{pid}")

    # --- billing_zip (planted — stored as Int64) ---
    billing_zips: list[int] = [rng.randint(0, 99999) for _ in range(NROWS)]

    # --- phone_number (planted) ---
    bad_phone_set = set(bad_phone_rows)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    phone_numbers: list[str] = []
    for i in range(NROWS):
        if i in bad_phone_set:
            # Insert one letter into area code digits
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

    # --- ship_date (planted) ---
    ship_dates: list[str] = []
    for i in range(NROWS):
        od = order_dates_raw[i]
        if i in ship_before_order_rows:
            sd = od - timedelta(days=rng.randint(1, 10))
        else:
            sd = od + timedelta(days=rng.randint(1, 14))
        ship_dates.append(sd.strftime("%Y-%m-%d"))

    # --- quantity (planted) ---
    neg_qty_set = set(neg_qty_rows)
    quantities: list[int] = []
    for i in range(NROWS):
        if i in neg_qty_set:
            quantities.append(rng.randint(-10, -1))
        else:
            quantities.append(rng.randint(1, 20))

    # --- discount_pct (planted) ---
    bad_discount_set = set(bad_discount_rows)
    discount_pcts: list[float] = []
    for i in range(NROWS):
        if i in bad_discount_set:
            discount_pcts.append(round(rng.uniform(101.0, 150.0), 2))
        else:
            discount_pcts.append(round(rng.uniform(0.0, 50.0), 2))

    # --- customer_name (planted) ---
    numeric_name_set = set(numeric_name_rows)
    customer_names: list[str] = []
    for i in range(NROWS):
        if i in numeric_name_set:
            idx = numeric_name_sorted.index(i)
            customer_names.append(NUMERIC_NAME_VALUES[idx])
        else:
            fn = rng.choice(FIRST_NAMES)
            ln = rng.choice(LAST_NAMES)
            customer_names.append(f"{fn} {ln}")

    # --- sku (planted — 25 duplicates) ---
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

    # --- rating (planted) ---
    bad_rating_set = set(bad_rating_rows)
    ratings: list[int] = [6 if i in bad_rating_set else rng.randint(1, 5) for i in range(NROWS)]

    # --- address_line1 / city / state (planted — 40 correlated nulls) ---
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
    # Clean columns (false-positive traps)                               #
    # ------------------------------------------------------------------ #

    # order_date — clean ISO date strings
    order_dates: list[str] = [d.strftime("%Y-%m-%d") for d in order_dates_raw]

    # free_text_notes — messy but valid
    free_text_notes_col: list[str] = [rng.choice(MESSY_NOTES) for _ in range(NROWS)]

    # product_description — varying length, HTML entities
    product_descriptions: list[str] = [rng.choice(PRODUCT_DESCRIPTIONS) for _ in range(NROWS)]

    # currency_code — low cardinality, valid
    currency_codes: list[str] = [rng.choice(CURRENCY_CODES) for _ in range(NROWS)]

    # user_agent — long, variable, valid
    user_agents: list[str] = [rng.choice(USER_AGENTS) for _ in range(NROWS)]

    # ip_address — mix IPv4 and IPv6
    ip_addresses: list[str] = []
    for _ in range(NROWS):
        if rng.random() < 0.7:
            ip_addresses.append(
                f"{rng.randint(1, 254)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
            )
        else:
            groups = [format(rng.randint(0, 0xFFFF), "x") for _ in range(8)]
            ip_addresses.append(":".join(groups))

    # tags — comma-separated, variable count
    tags_col: list[str] = [rng.choice(TAGS_POOL) for _ in range(NROWS)]

    # json_metadata — valid JSON strings
    json_metadata_col: list[str] = [rng.choice(JSON_METADATA_POOL) for _ in range(NROWS)]

    # phone_intl — mixed international formats, all valid
    phone_intl_col: list[str] = [rng.choice(INTL_PHONES) for _ in range(NROWS)]

    # address_line2 — 60% null (optional)
    address_line2s: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.60:
            address_line2s.append(None)
        else:
            address_line2s.append(rng.choice(["Apt 4B", "Suite 100", "Unit 7", "Floor 3", "Apt 12A", "Suite 200"]))

    # order_notes — 80% null (optional)
    order_notes_col: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.80:
            order_notes_col.append(None)
        else:
            order_notes_col.append(rng.choice(ORDER_NOTES_POOL))

    # referral_code — alphanumeric, valid
    code_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    referral_codes: list[str] = [
        "REF-" + "".join(rng.choice(code_chars) for _ in range(6))
        for _ in range(NROWS)
    ]

    # session_id — UUID v4 format, deterministic
    session_ids: list[str] = []
    for _ in range(NROWS):
        p1 = format(rng.randint(0, 0xFFFFFFFF), "08x")
        p2 = format(rng.randint(0, 0xFFFF), "04x")
        p3 = format((rng.randint(0, 0xFFFF) & 0x0FFF) | 0x4000, "04x")
        p4 = format((rng.randint(0, 0xFFFF) & 0x3FFF) | 0x8000, "04x")
        p5 = format(rng.randint(0, 0xFFFFFFFFFFFF), "012x")
        session_ids.append(f"{p1}-{p2}-{p3}-{p4}-{p5}")

    # created_at — ISO timestamps with UTC tz
    created_ats: list[str] = []
    for i in range(NROWS):
        od = order_dates_raw[i]
        dt = datetime(od.year, od.month, od.day,
                      rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59),
                      tzinfo=timezone.utc)
        created_ats.append(dt.isoformat())

    # updated_at — >= created_at
    updated_ats: list[str] = []
    for i in range(NROWS):
        od = order_dates_raw[i]
        extra_days = rng.randint(0, 30)
        dt = datetime(od.year, od.month, od.day,
                      rng.randint(0, 23), rng.randint(0, 59), 0,
                      tzinfo=timezone.utc) + timedelta(days=extra_days)
        updated_ats.append(dt.isoformat())

    # is_active — valid boolean strings
    is_active_col: list[str] = [rng.choice(["true", "false"]) for _ in range(NROWS)]

    # ------------------------------------------------------------------ #
    # Build Polars DataFrame (30 columns total)                          #
    # ------------------------------------------------------------------ #

    df = pl.DataFrame(
        {
            # Planted (15)
            "order_total":      order_totals,
            "customer_email":   customer_emails,
            "product_category": product_categories,
            "website_url":      website_urls,
            "billing_zip":      billing_zips,
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

    # ------------------------------------------------------------------ #
    # Build GroundTruth                                                   #
    # ------------------------------------------------------------------ #

    planted: dict[str, PlantedColumn] = {
        "order_total": PlantedColumn(
            issues=["outlier_values"],
            planted_count=20,
            description=(
                "20 rows have outlier order totals at 3.1 stddev above the mean (~255)."
            ),
            affected_rows=sorted(outlier_total_rows),
        ),
        "customer_email": PlantedColumn(
            issues=["invalid_format"],
            planted_count=200,
            description=(
                "200 rows have invalid email addresses (0.4% frequency — very low)."
            ),
            affected_rows=sorted(bad_email_rows),
        ),
        "product_category": PlantedColumn(
            issues=["distribution_shift"],
            planted_count=10_000,
            description=(
                "Last 10,000 rows (40000-49999) contain 3 new categories absent from "
                "the first 40K rows, representing data drift."
            ),
            affected_rows=list(range(drift_start, NROWS)),
        ),
        "website_url": PlantedColumn(
            issues=["wrong_type"],
            planted_count=15,
            description="15 rows contain email addresses instead of URL values.",
            affected_rows=sorted(bad_url_rows),
        ),
        "billing_zip": PlantedColumn(
            issues=["wrong_dtype"],
            planted_count=NROWS,
            description=(
                "billing_zip stored as Int64, silently dropping leading zeros "
                "(e.g. '07001' becomes 7001)."
            ),
            affected_rows=list(range(NROWS)),
        ),
        "phone_number": PlantedColumn(
            issues=["invalid_format"],
            planted_count=100,
            description=(
                "100 rows have letters mixed into the area code "
                "(e.g. '(5A5) 123-4567')."
            ),
            affected_rows=sorted(bad_phone_rows),
        ),
        "ship_date": PlantedColumn(
            issues=["logic_violation"],
            planted_count=30,
            description="30 rows have ship_date earlier than order_date.",
            affected_rows=sorted(ship_before_order_rows),
        ),
        "quantity": PlantedColumn(
            issues=["invalid_values"],
            planted_count=5,
            description="5 rows have negative quantity values.",
            affected_rows=sorted(neg_qty_rows),
        ),
        "discount_pct": PlantedColumn(
            issues=["out_of_range"],
            planted_count=8,
            description="8 rows have discount_pct > 100 (invalid percentage).",
            affected_rows=sorted(bad_discount_rows),
        ),
        "customer_name": PlantedColumn(
            issues=["wrong_type"],
            planted_count=12,
            description="12 rows have numeric characters substituted into customer names.",
            affected_rows=sorted(numeric_name_rows),
        ),
        "sku": PlantedColumn(
            issues=["duplicate_values"],
            planted_count=25,
            description=(
                "25 rows have duplicate SKU values (should be unique per product listing)."
            ),
            affected_rows=sorted(sku_dup_targets),
        ),
        "rating": PlantedColumn(
            issues=["enum_violation"],
            planted_count=8,
            description="8 rows have rating=6, outside the valid 1-5 range.",
            affected_rows=sorted(bad_rating_rows),
        ),
        "address_line1": PlantedColumn(
            issues=["null_values"],
            planted_count=40,
            description=(
                "40 rows have null address_line1 "
                "(correlated with city and state nulls)."
            ),
            affected_rows=sorted(null_address_rows),
        ),
        "city": PlantedColumn(
            issues=["null_values"],
            planted_count=40,
            description=(
                "40 rows have null city "
                "(correlated with address_line1 and state nulls)."
            ),
            affected_rows=sorted(null_address_rows),
        ),
        "state": PlantedColumn(
            issues=["null_values"],
            planted_count=40,
            description=(
                "40 rows have null state "
                "(correlated with address_line1 and city nulls)."
            ),
            affected_rows=sorted(null_address_rows),
        ),
    }

    clean = [
        "order_date", "free_text_notes", "product_description", "currency_code",
        "user_agent", "ip_address", "tags", "json_metadata", "phone_intl",
        "address_line2", "order_notes", "referral_code", "session_id",
        "created_at", "updated_at",
    ]

    total = sum(p.planted_count for p in planted.values())

    gt = GroundTruth(
        tier=2,
        version="1.0.0",
        rows=NROWS,
        columns=30,
        planted_columns=planted,
        clean_columns=clean,
        total_planted_issues=total,
    )

    return df, gt
