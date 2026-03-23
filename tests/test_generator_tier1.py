"""Tests for the Tier 1 dataset generator."""
from __future__ import annotations


def test_tier1_generation():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    assert len(df) == 5000
    assert len(df.columns) == 20
    assert gt.tier == 1
    assert len(gt.planted_columns) == 15
    assert len(gt.clean_columns) == 5


def test_tier1_deterministic():
    from dqbench.generator.tier1 import generate_tier1
    df1, gt1 = generate_tier1()
    df2, gt2 = generate_tier1()
    assert df1.equals(df2)
    assert gt1.total_planted_issues == gt2.total_planted_issues


def test_tier1_has_expected_columns():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    expected = {
        "customer_id", "first_name", "last_name", "email", "phone", "age", "income",
        "status", "signup_date", "last_login", "country", "zip_code",
        "shipping_address", "shipping_city", "shipping_zip",
        "order_count", "account_type", "last_updated", "notes", "referral_source",
    }
    assert set(df.columns) == expected


def test_tier1_planted_columns_match_spec():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    planted_names = set(gt.planted_columns.keys())
    expected_planted = {
        "customer_id", "first_name", "last_name", "email", "phone", "age", "income",
        "status", "signup_date", "last_login", "country", "zip_code",
        "shipping_address", "shipping_city", "shipping_zip",
    }
    assert planted_names == expected_planted


def test_tier1_clean_columns_match_spec():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    assert set(gt.clean_columns) == {
        "order_count", "account_type", "last_updated", "notes", "referral_source"
    }


def test_tier1_planted_counts():
    """Spot-check that planted_count values match the spec."""
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    pc = gt.planted_columns
    assert pc["customer_id"].planted_count == 15
    assert pc["first_name"].planted_count == 8
    assert pc["last_name"].planted_count == 5
    assert pc["email"].planted_count == 25
    assert pc["age"].planted_count == 28
    assert pc["income"].planted_count == 10
    assert pc["status"].planted_count == 15
    assert pc["signup_date"].planted_count == 12
    assert pc["last_login"].planted_count == 18
    assert pc["country"].planted_count == 10
    assert pc["shipping_address"].planted_count == 50
    assert pc["shipping_city"].planted_count == 50
    assert pc["shipping_zip"].planted_count == 50


def test_tier1_total_planted_issues():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    assert gt.total_planted_issues == sum(
        p.planted_count for p in gt.planted_columns.values()
    )


def test_tier1_affected_rows_within_bounds():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    for col, pc in gt.planted_columns.items():
        for r in pc.affected_rows:
            assert 0 <= r < 5000, f"Row {r} out of bounds for column {col}"


def test_tier1_first_name_nulls_present():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    null_count = df["first_name"].null_count()
    assert null_count == 8


def test_tier1_shipping_nulls_correlated():
    """Rows null in shipping_address must also be null in shipping_city and shipping_zip."""
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    addr_null = df["shipping_address"].is_null()
    city_null = df["shipping_city"].is_null()
    zip_null = df["shipping_zip"].is_null()
    assert (addr_null == city_null).all()
    assert (addr_null == zip_null).all()
    assert addr_null.sum() == 50


def test_tier1_income_outliers_present():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    outlier_count = (df["income"] == 9999999.99).sum()
    assert outlier_count == 10


def test_tier1_ground_truth_tier_and_version():
    from dqbench.generator.tier1 import generate_tier1
    df, gt = generate_tier1()
    assert gt.tier == 1
    assert gt.version == "1.0.0"
    assert gt.rows == 5000
    assert gt.columns == 20
