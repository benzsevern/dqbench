from dqbench.models import DQBenchFinding, TierResult, Scorecard


def test_finding_creation():
    f = DQBenchFinding(column="email", severity="error", check="format", message="bad")
    assert f.column == "email"
    assert f.confidence == 1.0


def test_tier_result():
    r = TierResult(tier=1, recall=0.9, precision=0.8, f1=0.847, false_positive_rate=0.1,
                   time_seconds=0.5, memory_mb=10, findings_count=50)
    assert r.f1 == 0.847


def test_scorecard():
    t1 = TierResult(tier=1, recall=1.0, precision=0.85, f1=0.92, false_positive_rate=0.0,
                    time_seconds=0.1, memory_mb=8, findings_count=45)
    t2 = TierResult(tier=2, recall=0.88, precision=0.72, f1=0.793, false_positive_rate=0.18,
                    time_seconds=1.4, memory_mb=45, findings_count=120)
    t3 = TierResult(tier=3, recall=0.65, precision=0.60, f1=0.624, false_positive_rate=0.22,
                    time_seconds=3.1, memory_mb=90, findings_count=200)
    sc = Scorecard(tool_name="Test", tool_version="1.0", tiers=[t1, t2, t3])
    # 0.92*0.2 + 0.793*0.4 + 0.624*0.4 = 0.184 + 0.3172 + 0.2496 = 0.7508
    assert abs(sc.dqbench_score - 75.08) < 0.1
