"""Tests for the benchmark runner and report."""
from __future__ import annotations
import io
import json
from pathlib import Path

from dqbench.runner import run_benchmark
from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding, Scorecard, TierResult


class NullAdapter(DQBenchAdapter):
    @property
    def name(self) -> str:
        return "NullTool"

    @property
    def version(self) -> str:
        return "0.0"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        return []


def test_run_benchmark_returns_scorecard():
    scorecard = run_benchmark(NullAdapter(), tiers=[1])
    assert scorecard.tool_name == "NullTool"
    assert len(scorecard.tiers) == 1
    assert scorecard.tiers[0].recall == 0.0  # null adapter finds nothing


def test_run_benchmark_null_adapter_zero_precision():
    scorecard = run_benchmark(NullAdapter(), tiers=[1])
    tier = scorecard.tiers[0]
    assert tier.precision == 0.0
    assert tier.f1 == 0.0


def test_run_benchmark_multiple_tiers():
    scorecard = run_benchmark(NullAdapter(), tiers=[1, 2])
    assert len(scorecard.tiers) == 2
    assert scorecard.tiers[0].tier == 1
    assert scorecard.tiers[1].tier == 2


def test_run_benchmark_all_tiers():
    scorecard = run_benchmark(NullAdapter())
    assert len(scorecard.tiers) == 3
    for t in scorecard.tiers:
        assert t.recall == 0.0


def test_scorecard_has_timing():
    scorecard = run_benchmark(NullAdapter(), tiers=[1])
    tier = scorecard.tiers[0]
    assert tier.time_seconds >= 0.0
    assert tier.memory_mb >= 0.0


def test_report_json_structure():
    from dqbench.report import report_json
    t1 = TierResult(
        tier=1, recall=0.0, precision=0.0, f1=0.0, false_positive_rate=0.0,
        issue_recall=0.0, issue_precision=0.0, issue_f1=0.0,
        time_seconds=0.1, memory_mb=5.0, findings_count=0,
    )
    sc = Scorecard(tool_name="NullTool", tool_version="0.0", tiers=[t1])
    buf = io.StringIO()
    report_json(sc, buf)
    data = json.loads(buf.getvalue())
    assert data["tool_name"] == "NullTool"
    assert data["dqbench_score"] == 0.0
    assert len(data["tiers"]) == 1
    assert data["tiers"][0]["tier"] == 1


def test_report_rich_runs_without_error():
    from dqbench.report import report_rich
    t1 = TierResult(
        tier=1, recall=0.5, precision=0.5, f1=0.5, false_positive_rate=0.1,
        issue_recall=0.4, issue_precision=0.4, issue_f1=0.4,
        time_seconds=0.5, memory_mb=10.0, findings_count=20,
    )
    sc = Scorecard(tool_name="TestTool", tool_version="1.0", tiers=[t1])
    # Should run without raising
    report_rich(sc)
