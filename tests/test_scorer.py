from dqbench.scorer import score_tier, _finding_matches_issue
from dqbench.ground_truth import GroundTruth, PlantedColumn
from dqbench.models import DQBenchFinding


def _make_gt(planted: dict[str, list[str]], clean: list[str], tier: int = 1) -> GroundTruth:
    return GroundTruth(
        tier=tier,
        version="1.0",
        rows=100,
        columns=len(planted) + len(clean),
        planted_columns={
            k: PlantedColumn(issues=v, planted_count=1, description="test")
            for k, v in planted.items()
        },
        clean_columns=clean,
        total_planted_issues=sum(len(v) for v in planted.values()),
    )


# ---------------------------------------------------------------------------
# Column-level tests (existing)
# ---------------------------------------------------------------------------

def test_perfect_score():
    gt = _make_gt({"email": ["format"], "age": ["range"]}, ["notes"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="format", message="bad"),
        DQBenchFinding(column="age", severity="warning", check="range", message="outlier"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.false_positive_rate == 0.0


def test_false_positive():
    gt = _make_gt({"email": ["format"]}, ["notes", "tags"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="x", message="x"),
        DQBenchFinding(column="notes", severity="warning", check="x", message="x"),  # FP
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0
    assert result.precision == 0.5   # 1 TP, 1 FP
    assert result.false_positive_rate == 0.5  # 1 of 2 clean cols flagged


def test_info_on_clean_not_fp():
    gt = _make_gt({"email": ["format"]}, ["notes"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="x", message="x"),
        DQBenchFinding(column="notes", severity="info", check="x", message="x"),  # NOT a FP
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.false_positive_rate == 0.0


def test_missed_column():
    gt = _make_gt({"email": ["format"], "age": ["range"]}, [])
    findings = [DQBenchFinding(column="email", severity="error", check="x", message="x")]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 0.5


def test_comma_joined_column():
    gt = _make_gt({"start": ["temporal"], "end": ["temporal"]}, [])
    findings = [
        DQBenchFinding(column="end,start", severity="warning", check="temporal", message="x")
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0  # both columns detected via comma split


# ---------------------------------------------------------------------------
# Issue-level keyword matching unit tests
# ---------------------------------------------------------------------------

def test_keyword_match_exact_check():
    """Finding with exact check match → issue detected."""
    finding = DQBenchFinding(column="email", severity="error", check="uniqueness", message="x")
    assert _finding_matches_issue(finding, "uniqueness") is True


def test_keyword_match_in_message():
    """Finding with keyword in message → issue detected."""
    finding = DQBenchFinding(column="email", severity="error", check="generic_check",
                             message="found duplicate values in this column")
    assert _finding_matches_issue(finding, "uniqueness") is True


def test_keyword_match_in_check_field():
    """Finding with keyword in check field (not exact) → issue detected."""
    finding = DQBenchFinding(column="email", severity="error", check="duplicate_detection",
                             message="some message")
    assert _finding_matches_issue(finding, "uniqueness") is True


def test_wrong_check_type_no_issue_match():
    """Finding on planted column but wrong check type → column detected, issue NOT detected."""
    gt = _make_gt({"npi_number": ["luhn"]}, ["notes"])
    # nullability finding on a luhn column — column is detected but issue is NOT
    findings = [
        DQBenchFinding(column="npi_number", severity="warning", check="nullability",
                       message="null values found"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    # Column-level: detected
    assert result.recall == 1.0
    # Issue-level: NOT detected (nullability != luhn)
    assert result.issue_recall == 0.0


def test_generic_nullability_on_luhn_column_not_issue_match():
    """Generic nullability finding on column with 'luhn' issue → issue NOT detected."""
    finding = DQBenchFinding(column="npi_number", severity="warning", check="nullability",
                             message="8% of values are null")
    assert _finding_matches_issue(finding, "luhn") is False


def test_luhn_keyword_in_message_matches_luhn_issue():
    """Finding with 'luhn' in message matches luhn issue."""
    finding = DQBenchFinding(column="npi_number", severity="error", check="format_check",
                             message="NPI fails Luhn check digit validation")
    assert _finding_matches_issue(finding, "luhn") is True


def test_encoding_keyword_matches_encoding_issue():
    """Finding mentioning encoding matches encoding issue type."""
    finding = DQBenchFinding(column="claim_notes", severity="warning", check="encoding_check",
                             message="detected latin-1 characters in utf-8 field")
    assert _finding_matches_issue(finding, "encoding") is True


def test_cross_column_keyword_matches():
    """Finding with mismatch keyword matches cross_column issue."""
    finding = DQBenchFinding(column="patient_state", severity="error", check="cross_column",
                             message="state doesn't match zip prefix")
    assert _finding_matches_issue(finding, "cross_column") is True


def test_semantic_keyword_matches():
    """Finding mentioning weekend matches semantic issue."""
    finding = DQBenchFinding(column="service_day", severity="warning", check="business_rule",
                             message="service scheduled on a weekend for weekday-only provider")
    assert _finding_matches_issue(finding, "semantic") is True


# ---------------------------------------------------------------------------
# Issue-level scoring integration tests
# ---------------------------------------------------------------------------

def test_issue_recall_perfect():
    """All issues specifically detected → issue_recall = 1.0."""
    gt = _make_gt({"email": ["invalid_format"], "age": ["outlier_values"]}, [])
    findings = [
        DQBenchFinding(column="email", severity="error", check="format",
                       message="found invalid email format"),
        DQBenchFinding(column="age", severity="warning", check="outlier",
                       message="extreme outlier values detected"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.issue_recall == 1.0
    assert result.issue_f1 > 0.0


def test_issue_recall_zero_when_generic_only():
    """Only generic checks fire, no specific issue matches → issue_recall = 0.0."""
    gt = _make_gt({"npi_number": ["luhn"], "patient_state": ["cross_column"]}, [])
    findings = [
        # Generic profiler fires — column detected but issue not matched
        DQBenchFinding(column="npi_number", severity="warning", check="nullability",
                       message="has null values"),
        DQBenchFinding(column="patient_state", severity="warning", check="cardinality",
                       message="low cardinality categorical column"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    # Column recall is 100% (both columns flagged)
    assert result.recall == 1.0
    # But issue recall is 0% (no specific issues caught)
    assert result.issue_recall == 0.0


def test_issue_f1_present_in_result():
    """TierResult has issue_f1 field populated."""
    gt = _make_gt({"email": ["invalid_format"]}, ["notes"])
    findings = [
        DQBenchFinding(column="email", severity="error", check="format", message="bad email"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert hasattr(result, "issue_recall")
    assert hasattr(result, "issue_precision")
    assert hasattr(result, "issue_f1")
    assert 0.0 <= result.issue_recall <= 1.0
    assert 0.0 <= result.issue_precision <= 1.0
    assert 0.0 <= result.issue_f1 <= 1.0


def test_partial_issue_detection():
    """Only one of two planted issues detected → issue_recall = 0.5."""
    gt = _make_gt({"email": ["invalid_format"], "age": ["outlier_values"]}, [])
    findings = [
        # email issue detected specifically
        DQBenchFinding(column="email", severity="error", check="format",
                       message="invalid email format"),
        # age column detected but with wrong check type
        DQBenchFinding(column="age", severity="warning", check="nullability",
                       message="age has null values"),
    ]
    result = score_tier(findings, gt, tier=1, time_seconds=0.1, memory_mb=5)
    assert result.recall == 1.0          # both columns found
    assert result.issue_recall == 0.5    # only 1 of 2 issues specifically caught
