"""Tests for ER and Pipeline data models."""
from dqbench.models import (
    ERTierResult, ERScorecard, ERRealResult,
    PipelineTierResult, PipelineScorecard,
)


def test_er_scorecard_composite():
    tiers = [
        ERTierResult(tier=1, precision=1.0, recall=1.0, f1=1.0, false_positives=0, false_negatives=0, time_seconds=0.1, memory_mb=10),
        ERTierResult(tier=2, precision=0.8, recall=0.9, f1=0.85, false_positives=5, false_negatives=3, time_seconds=0.5, memory_mb=50),
        ERTierResult(tier=3, precision=0.7, recall=0.6, f1=0.65, false_positives=10, false_negatives=8, time_seconds=1.0, memory_mb=100),
    ]
    sc = ERScorecard(tool_name="test", tool_version="1.0", tiers=tiers)
    expected = round(1.0 * 0.20 * 100 + 0.85 * 0.40 * 100 + 0.65 * 0.40 * 100, 2)
    assert sc.dqbench_er_score == expected


def test_pipeline_scorecard_composite():
    tiers = [
        PipelineTierResult(tier=1, transform_accuracy=0.9, dedup_accuracy=1.0, composite=0.94, output_rows=90, expected_rows=90, time_seconds=0.1, memory_mb=10),
        PipelineTierResult(tier=2, transform_accuracy=0.8, dedup_accuracy=0.9, composite=0.84, output_rows=420, expected_rows=425, time_seconds=0.5, memory_mb=50),
        PipelineTierResult(tier=3, transform_accuracy=0.7, dedup_accuracy=0.8, composite=0.74, output_rows=790, expected_rows=800, time_seconds=1.0, memory_mb=100),
    ]
    sc = PipelineScorecard(tool_name="test", tool_version="1.0", tiers=tiers)
    expected = round(0.94 * 0.20 * 100 + 0.84 * 0.40 * 100 + 0.74 * 0.40 * 100, 2)
    assert sc.dqbench_pipeline_score == expected


def test_er_real_result():
    r = ERRealResult(dataset_name="dblp-acm", precision=0.95, recall=0.92, f1=0.935, time_seconds=2.0)
    assert r.dataset_name == "dblp-acm"
    assert r.f1 == 0.935
