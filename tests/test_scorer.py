from dqbench.scorer import score_tier
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
