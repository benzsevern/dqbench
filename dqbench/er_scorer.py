"""Scoring logic for ER benchmarks."""
from __future__ import annotations

from dqbench.models import ERTierResult
from dqbench.er_ground_truth import ERGroundTruth


def _normalize_pairs(pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Normalize pairs to (min, max) for symmetric matching."""
    return {(min(a, b), max(a, b)) for a, b in pairs}


def score_er_tier(
    predictions: list[tuple[int, int]],
    ground_truth: ERGroundTruth,
    tier: int,
    time_seconds: float,
    memory_mb: float,
) -> ERTierResult:
    """Score ER predictions against ground truth using pair-level P/R/F1."""
    true_pairs = _normalize_pairs(ground_truth.duplicate_pairs)
    pred_pairs = _normalize_pairs(predictions)

    true_positives = pred_pairs & true_pairs
    false_positives = pred_pairs - true_pairs
    false_negatives = true_pairs - pred_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    if tp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    return ERTierResult(
        tier=tier,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=fp,
        false_negatives=fn,
        time_seconds=time_seconds,
        memory_mb=memory_mb,
    )
