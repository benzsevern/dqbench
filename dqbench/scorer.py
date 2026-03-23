"""Compute benchmark scores from findings vs ground truth."""
from __future__ import annotations
from dqbench.models import DQBenchFinding, TierResult
from dqbench.ground_truth import GroundTruth


def score_tier(
    findings: list[DQBenchFinding],
    ground_truth: GroundTruth,
    tier: int,
    time_seconds: float,
    memory_mb: float,
) -> TierResult:
    planted_cols = set(ground_truth.planted_columns.keys())
    clean_cols = set(ground_truth.clean_columns)

    # Detection: any finding on a planted column (any severity)
    detected_cols = set()
    for f in findings:
        for part in f.column.split(","):
            col = part.strip()
            if col in planted_cols:
                detected_cols.add(col)

    # False positives: WARNING/ERROR on clean columns (INFO is NOT a FP)
    fp_cols = set()
    for f in findings:
        if f.severity.lower() in ("error", "warning"):
            for part in f.column.split(","):
                col = part.strip()
                if col in clean_cols:
                    fp_cols.add(col)

    tp = len(detected_cols)
    fp = len(fp_cols)

    recall = tp / len(planted_cols) if planted_cols else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = len(fp_cols) / len(clean_cols) if clean_cols else 0.0

    return TierResult(
        tier=tier,
        recall=round(recall, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        time_seconds=round(time_seconds, 3),
        memory_mb=round(memory_mb, 1),
        findings_count=len(findings),
    )
