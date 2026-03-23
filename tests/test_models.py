from dqbench.models import DQBenchFinding, TierResult, Scorecard


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
