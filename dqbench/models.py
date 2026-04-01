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
    # Column-level (existing)
    recall: float
    precision: float
    f1: float
    false_positive_rate: float
    # Meta
    time_seconds: float
    memory_mb: float
    findings_count: int
    # Issue-level
    issue_recall: float = 0.0
    issue_precision: float = 0.0
    issue_f1: float = 0.0


@dataclass
class Scorecard:
    tool_name: str
    tool_version: str
    tiers: list[TierResult]
    llm_cost: float | None = None
    confidence_calibration: float | None = None

    @property
    def dqbench_score(self) -> float:
        """Composite score using issue-level F1 (measures targeted detection)."""
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        total = 0.0
        for t in self.tiers:
            total += t.issue_f1 * weights.get(t.tier, 0) * 100
        return round(total, 2)


@dataclass
class TransformColumnResult:
    column: str
    planted_cells: int
    correct_cells: int
    wrong_cells: int
    skipped_cells: int  # cells where output == messy input
    accuracy: float


@dataclass
class TransformTierResult:
    tier: int
    accuracy: float
    correct_cells: int
    wrong_cells: int
    skipped_cells: int
    planted_cells: int
    time_seconds: float
    memory_mb: float
    per_column: list[TransformColumnResult]


@dataclass
class TransformScorecard:
    tool_name: str
    tool_version: str
    tiers: list[TransformTierResult]

    @property
    def composite_score(self) -> float:
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        total = 0.0
        for t in self.tiers:
            total += t.accuracy * weights.get(t.tier, 0) * 100
        return round(total, 2)


@dataclass
class ERTierResult:
    tier: int
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int
    time_seconds: float
    memory_mb: float


@dataclass
class ERRealResult:
    dataset_name: str
    precision: float
    recall: float
    f1: float
    time_seconds: float


@dataclass
class ERScorecard:
    tool_name: str
    tool_version: str
    tiers: list[ERTierResult]
    real_datasets: list[ERRealResult] | None = None

    @property
    def dqbench_er_score(self) -> float:
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.f1 * weights.get(t.tier, 0) * 100 for t in self.tiers), 2)


@dataclass
class PipelineTierResult:
    tier: int
    transform_accuracy: float
    dedup_accuracy: float
    composite: float
    output_rows: int
    expected_rows: int
    time_seconds: float
    memory_mb: float


@dataclass
class PipelineScorecard:
    tool_name: str
    tool_version: str
    tiers: list[PipelineTierResult]

    @property
    def dqbench_pipeline_score(self) -> float:
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.composite * weights.get(t.tier, 0) * 100 for t in self.tiers), 2)


@dataclass
class OCRCompanyPrediction:
    record_id: str
    confidence: float
    weakest_token: str = ""
    suggested_correction: str = ""
    review_required: bool | None = None


@dataclass
class OCRCompanyTierResult:
    tier: int
    confidence_separation: float
    clean_flag_rate: float
    corrupted_flag_rate: float
    weakest_token_hit_rate: float
    suggestion_coverage_rate: float
    suggestion_exact_hit_rate: float
    suggestion_improvement_rate: float
    avg_similarity_delta_on_suggestions: float
    composite: float
    rows: int
    time_seconds: float
    memory_mb: float


@dataclass
class OCRCompanyScorecard:
    tool_name: str
    tool_version: str
    tiers: list[OCRCompanyTierResult]

    @property
    def dqbench_ocr_company_score(self) -> float:
        weights = {1: 0.20, 2: 0.40, 3: 0.40}
        return round(sum(t.composite * weights.get(t.tier, 0) * 100 for t in self.tiers), 2)
