"""Tests for the Tier 2 dataset generator."""
from __future__ import annotations


def test_tier2_generation():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert len(df) == 50_000
    assert len(df.columns) == 30
    assert gt.tier == 2
    assert len(gt.planted_columns) == 15
    assert len(gt.clean_columns) == 15


def test_tier2_deterministic():
    from dqbench.generator.tier2 import generate_tier2
    df1, gt1 = generate_tier2()
    df2, gt2 = generate_tier2()
    assert df1.equals(df2)
    assert gt1.total_planted_issues == gt2.total_planted_issues


def test_tier2_has_expected_columns():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    expected_planted = {
        "order_total", "customer_email", "product_category", "website_url",
        "billing_zip", "phone_number", "ship_date", "quantity", "discount_pct",
        "customer_name", "sku", "rating", "address_line1", "city", "state",
    }
    expected_clean = {
        "order_date", "free_text_notes", "product_description", "currency_code",
        "user_agent", "ip_address", "tags", "json_metadata", "phone_intl",
        "address_line2", "order_notes", "referral_code", "session_id",
        "created_at", "updated_at",
    }
    assert set(df.columns) == expected_planted | expected_clean


def test_tier2_planted_columns_match_spec():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    expected_planted = {
        "order_total", "customer_email", "product_category", "website_url",
        "billing_zip", "phone_number", "ship_date", "quantity", "discount_pct",
        "customer_name", "sku", "rating", "address_line1", "city", "state",
    }
    assert set(gt.planted_columns.keys()) == expected_planted


def test_tier2_clean_columns_match_spec():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    expected_clean = {
        "order_date", "free_text_notes", "product_description", "currency_code",
        "user_agent", "ip_address", "tags", "json_metadata", "phone_intl",
        "address_line2", "order_notes", "referral_code", "session_id",
        "created_at", "updated_at",
    }
    assert set(gt.clean_columns) == expected_clean


def test_tier2_planted_counts():
    from dqbench.generator.tier2 import generate_tier2, NROWS
    df, gt = generate_tier2()
    pc = gt.planted_columns
    assert pc["order_total"].planted_count == 20
    assert pc["customer_email"].planted_count == 200
    assert pc["product_category"].planted_count == 10_000
    assert pc["website_url"].planted_count == 15
    assert pc["billing_zip"].planted_count == NROWS
    assert pc["phone_number"].planted_count == 100
    assert pc["ship_date"].planted_count == 30
    assert pc["quantity"].planted_count == 5
    assert pc["discount_pct"].planted_count == 8
    assert pc["customer_name"].planted_count == 12
    assert pc["sku"].planted_count == 25
    assert pc["rating"].planted_count == 8
    assert pc["address_line1"].planted_count == 40
    assert pc["city"].planted_count == 40
    assert pc["state"].planted_count == 40


def test_tier2_total_planted_issues():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert gt.total_planted_issues == sum(
        p.planted_count for p in gt.planted_columns.values()
    )


def test_tier2_affected_rows_within_bounds():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    for col, pc in gt.planted_columns.items():
        for r in pc.affected_rows:
            assert 0 <= r < 50_000, f"Row {r} out of bounds for column {col}"


def test_tier2_billing_zip_is_int():
    """billing_zip must be stored as an integer type (leading zeros lost)."""
    from dqbench.generator.tier2 import generate_tier2
    import polars as pl
    df, gt = generate_tier2()
    assert df["billing_zip"].dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Int16)


def test_tier2_rating_outliers():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert (df["rating"] == 6).sum() == 8


def test_tier2_negative_quantities():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert (df["quantity"] < 0).sum() == 5


def test_tier2_discount_over_100():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert (df["discount_pct"] > 100).sum() == 8


def test_tier2_address_nulls_correlated():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    addr_null = df["address_line1"].is_null()
    city_null = df["city"].is_null()
    state_null = df["state"].is_null()
    assert (addr_null == city_null).all()
    assert (addr_null == state_null).all()
    assert addr_null.sum() == 40


def test_tier2_product_category_drift():
    """First 40K rows must not contain drift categories; last 10K must contain them."""
    from dqbench.generator.tier2 import generate_tier2, BASE_CATEGORIES, DRIFT_CATEGORIES
    df, gt = generate_tier2()
    first_40k = df["product_category"].head(40_000)
    last_10k = df["product_category"].tail(10_000)
    assert all(c in BASE_CATEGORIES for c in first_40k.to_list())
    assert any(c in DRIFT_CATEGORIES for c in last_10k.to_list())


def test_tier2_ship_before_order_count():
    """30 rows must have ship_date < order_date."""
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    bad = (df["ship_date"] < df["order_date"]).sum()
    assert bad == 30


def test_tier2_sku_duplicates():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    sku_counts = df["sku"].value_counts()
    duplicated_skus = sku_counts.filter(sku_counts["count"] > 1)
    # Each dup target takes source's SKU — so 25 target rows each duplicate a source
    total_dup_rows = (duplicated_skus["count"] - 1).sum()
    assert total_dup_rows == 25


def test_tier2_ground_truth_metadata():
    from dqbench.generator.tier2 import generate_tier2
    df, gt = generate_tier2()
    assert gt.tier == 2
    assert gt.version == "1.0.0"
    assert gt.rows == 50_000
    assert gt.columns == 30
