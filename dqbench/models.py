"""DQBench data models."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DQBenchFinding:
    column: str
    severity: str
    check: str
    message: str
    confidence: float = 1.0


@dataclass
class TierResult:
    tier: int
    recall: float
    precision: float
    f1: float
    false_positive_rate: float
    time_seconds: float
    memory_mb: float
    findings_count: int


@dataclass
class Scorecard:
    tool_name: str
    tool_version: str
    tiers: list[TierResult]
    llm_cost: float | None = None
    confidence_calibration: float | None = None

    @property
    def dqbench_score(self) -> float:
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        total = 0.0
        for t in self.tiers:
            total += t.f1 * weights.get(t.tier, 0) * 100
        return round(total, 2)
