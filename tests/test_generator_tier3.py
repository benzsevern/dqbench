"""Tests for the Tier 3 dataset generator."""
from __future__ import annotations


def test_tier3_generation():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    assert len(df) == 100_000
    assert len(df.columns) == 50
    assert gt.tier == 3
    assert len(gt.planted_columns) == 25
    assert len(gt.clean_columns) == 25


def test_tier3_deterministic():
    from dqbench.generator.tier3 import generate_tier3
    df1, gt1 = generate_tier3()
    df2, gt2 = generate_tier3()
    assert df1.equals(df2)
    assert gt1.total_planted_issues == gt2.total_planted_issues


def test_tier3_has_planted_columns():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    expected_planted = {
        "npi_number", "patient_state", "service_date", "claim_notes",
        "provider_name", "diagnosis_desc", "claim_amount", "service_day",
        "submit_date", "record_number", "patient_age", "procedure_code",
        "insurance_id", "patient_zip", "dosage_amount", "lab_result",
        "admission_date", "discharge_date", "primary_dx", "secondary_dx",
        "charge_code", "patient_name", "referring_npi", "auth_number",
        "payment_amount",
    }
    assert set(gt.planted_columns.keys()) == expected_planted


def test_tier3_has_clean_columns():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    expected_clean = {
        "patient_id", "date_of_birth", "gender", "policy_max_amount",
        "provider_type", "facility_code", "billing_code", "modifier_code",
        "place_of_service", "revenue_code", "drg_code", "attending_physician",
        "referral_source", "patient_phone", "patient_email", "emergency_contact",
        "primary_insurance", "secondary_insurance", "copay_amount", "deductible_met",
        "prior_auth_flag", "service_type", "claim_status", "adjudication_date",
        "remittance_amount",
    }
    assert set(gt.clean_columns) == expected_clean


def test_tier3_planted_counts():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    pc = gt.planted_columns
    assert pc["npi_number"].planted_count == 50
    assert pc["patient_state"].planted_count == 30
    assert pc["service_date"].planted_count == 20
    assert pc["claim_notes"].planted_count == 15
    assert pc["provider_name"].planted_count == 10
    assert pc["diagnosis_desc"].planted_count == 25
    assert pc["claim_amount"].planted_count == 12
    assert pc["service_day"].planted_count == 20
    assert pc["submit_date"].planted_count == 15
    assert pc["record_number"].planted_count == 8
    assert pc["patient_age"].planted_count == 5
    assert pc["procedure_code"].planted_count == 18
    assert pc["insurance_id"].planted_count == 22
    assert pc["patient_zip"].planted_count == 10
    assert pc["dosage_amount"].planted_count == 8
    assert pc["lab_result"].planted_count == 6
    assert pc["admission_date"].planted_count == 12
    assert pc["discharge_date"].planted_count == 12
    assert pc["primary_dx"].planted_count == 15
    assert pc["secondary_dx"].planted_count == 10
    assert pc["charge_code"].planted_count == 20
    assert pc["patient_name"].planted_count == 8
    assert pc["referring_npi"].planted_count == 25
    assert pc["auth_number"].planted_count == 5
    assert pc["payment_amount"].planted_count == 7


def test_tier3_total_planted_issues():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    assert gt.total_planted_issues == sum(
        p.planted_count for p in gt.planted_columns.values()
    )


def test_tier3_affected_rows_within_bounds():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    for col, pc in gt.planted_columns.items():
        for r in pc.affected_rows:
            assert 0 <= r < 100_000, f"Row {r} out of bounds for column {col}"


def test_tier3_luhn_failures():
    """npi_number: exactly 50 rows fail Luhn; the rest pass."""
    from dqbench.generator.tier3 import generate_tier3, luhn_checksum
    df, gt = generate_tier3()
    npi_col = df["npi_number"].to_list()
    fail_count = sum(1 for n in npi_col if not luhn_checksum(n))
    assert fail_count == 50


def test_tier3_referring_npi_luhn_failures():
    from dqbench.generator.tier3 import generate_tier3, luhn_checksum
    df, gt = generate_tier3()
    ref_npi_col = df["referring_npi"].to_list()
    fail_count = sum(1 for n in ref_npi_col if not luhn_checksum(n))
    assert fail_count == 25


def test_tier3_negative_payments():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    assert (df["payment_amount"] < 0).sum() == 7


def test_tier3_claim_exceeds_policy():
    """12 rows must have claim_amount > policy_max_amount."""
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    exceed = (df["claim_amount"] > df["policy_max_amount"]).sum()
    assert exceed == 12


def test_tier3_admission_after_discharge():
    """12 rows must have admission_date > discharge_date."""
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    bad = (df["admission_date"] > df["discharge_date"]).sum()
    assert bad == 12


def test_tier3_submit_before_service():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    bad = (df["submit_date"] < df["service_date"]).sum()
    assert bad == 15


def test_tier3_patient_zip_nonexistent():
    from dqbench.generator.tier3 import generate_tier3, NONEXISTENT_ZIPS
    df, gt = generate_tier3()
    bad = df["patient_zip"].is_in(NONEXISTENT_ZIPS).sum()
    assert bad == 10


def test_tier3_charge_code_duplicates():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    cc_counts = df["charge_code"].value_counts()
    total_dup_rows = (cc_counts["count"] - 1).sum()
    assert total_dup_rows == 20


def test_tier3_lab_result_bimodal_gap():
    """6 lab_result values should fall in the bimodal gap (42-58)."""
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    gap = ((df["lab_result"] >= 42.0) & (df["lab_result"] <= 58.0)).sum()
    assert gap == 6


def test_tier3_record_number_gaps():
    """record_number should have exactly 8 gaps in the sequence."""
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    rn = df["record_number"].to_list()
    gaps = sum(1 for a, b in zip(rn, rn[1:]) if b - a > 1)
    assert gaps == 8


def test_tier3_smart_quotes_in_diagnosis():
    """25 diagnosis_desc values should contain curly/smart quotes."""
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    smart_chars = {"\u2018", "\u2019", "\u201c", "\u201d"}
    count = sum(
        1 for s in df["diagnosis_desc"].to_list()
        if s and any(c in s for c in smart_chars)
    )
    assert count == 25


def test_tier3_provider_name_zero_width():
    """10 provider_name values should contain zero-width chars."""
    from dqbench.generator.tier3 import generate_tier3, ZERO_WIDTH_CHARS
    df, gt = generate_tier3()
    zwc_set = set(ZERO_WIDTH_CHARS)
    count = sum(
        1 for s in df["provider_name"].to_list()
        if s and any(c in s for c in zwc_set)
    )
    assert count == 10


def test_tier3_ground_truth_metadata():
    from dqbench.generator.tier3 import generate_tier3
    df, gt = generate_tier3()
    assert gt.tier == 3
    assert gt.version == "1.0.0"
    assert gt.rows == 100_000
    assert gt.columns == 50
