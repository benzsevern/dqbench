from dqbench.models import DQBenchFinding, TierResult, Scorecard, OCRCompanyTierResult, OCRCompanyScorecard


def test_finding_creation():
    f = DQBenchFinding(column="email", severity="error", check="format", message="bad")
    assert f.column == "email"
    assert f.confidence == 1.0


def test_tier_result():
    r = TierResult(
        tier=1, recall=0.9, precision=0.8, f1=0.847, false_positive_rate=0.1,
        issue_recall=0.7, issue_precision=0.65, issue_f1=0.674,
        time_seconds=0.5, memory_mb=10, findings_count=50,
    )
    assert r.f1 == 0.847
    assert r.issue_f1 == 0.674


def test_scorecard():
    t1 = TierResult(
        tier=1, recall=1.0, precision=0.85, f1=0.92, false_positive_rate=0.0,
        issue_recall=0.8, issue_precision=0.75, issue_f1=0.774,
        time_seconds=0.1, memory_mb=8, findings_count=45,
    )
    t2 = TierResult(
        tier=2, recall=0.88, precision=0.72, f1=0.793, false_positive_rate=0.18,
        issue_recall=0.6, issue_precision=0.55, issue_f1=0.574,
        time_seconds=1.4, memory_mb=45, findings_count=120,
    )
    t3 = TierResult(
        tier=3, recall=0.65, precision=0.60, f1=0.624, false_positive_rate=0.22,
        issue_recall=0.4, issue_precision=0.38, issue_f1=0.390,
        time_seconds=3.1, memory_mb=90, findings_count=200,
    )
    sc = Scorecard(tool_name="Test", tool_version="1.0", tiers=[t1, t2, t3])
    # DQBench Score now uses issue_f1: 0.774*0.2 + 0.574*0.4 + 0.390*0.4 = 0.1548 + 0.2296 + 0.156 = 0.5404
    expected = round((0.774 * 0.20 + 0.574 * 0.40 + 0.390 * 0.40) * 100, 2)
    assert abs(sc.dqbench_score - expected) < 0.1


def test_ocr_company_scorecard():
    t1 = OCRCompanyTierResult(
        tier=1,
        confidence_separation=0.5,
        clean_flag_rate=0.1,
        corrupted_flag_rate=0.9,
        weakest_token_hit_rate=0.8,
        suggestion_coverage_rate=0.6,
        suggestion_exact_hit_rate=0.7,
        suggestion_improvement_rate=0.75,
        avg_similarity_delta_on_suggestions=0.02,
        composite=0.78,
        rows=100,
        time_seconds=0.2,
        memory_mb=5.0,
    )
    t2 = OCRCompanyTierResult(
        tier=2,
        confidence_separation=0.6,
        clean_flag_rate=0.12,
        corrupted_flag_rate=0.88,
        weakest_token_hit_rate=0.82,
        suggestion_coverage_rate=0.58,
        suggestion_exact_hit_rate=0.72,
        suggestion_improvement_rate=0.77,
        avg_similarity_delta_on_suggestions=0.03,
        composite=0.80,
        rows=150,
        time_seconds=0.4,
        memory_mb=7.0,
    )
    sc = OCRCompanyScorecard(tool_name="OCR", tool_version="1.0", tiers=[t1, t2])
    expected = round((0.78 * 0.20 + 0.80 * 0.40) * 100, 2)
    assert abs(sc.dqbench_ocr_company_score - expected) < 0.1
