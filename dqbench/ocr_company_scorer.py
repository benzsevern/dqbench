"""Score OCR company benchmark predictions against benchmark truth."""
from __future__ import annotations

from difflib import SequenceMatcher

import polars as pl

from dqbench.models import OCRCompanyPrediction, OCRCompanyTierResult


def score_ocr_company_tier(
    predictions: list[OCRCompanyPrediction],
    dataset: pl.DataFrame,
    tier: int,
    time_seconds: float,
    memory_mb: float,
    threshold: float = 0.65,
) -> OCRCompanyTierResult:
    rows = dataset.to_dicts()
    prediction_by_id = {prediction.record_id: prediction for prediction in predictions}

    clean_confidences: list[float] = []
    corrupted_confidences: list[float] = []
    clean_flags = 0
    corrupted_flags = 0
    weakest_hits = 0
    suggestion_rows = 0
    suggestion_exact_hits = 0
    suggestion_improvements = 0
    similarity_deltas: list[float] = []

    for row in rows:
        prediction = prediction_by_id.get(str(row["record_id"]))
        confidence = clamp_confidence(prediction.confidence if prediction else 1.0)
        review_required = (
            prediction.review_required if prediction and prediction.review_required is not None
            else confidence < threshold
        )
        weakest_token = prediction.weakest_token if prediction else ""
        suggestion = prediction.suggested_correction if prediction else ""

        corrupted = normalize(str(row["company_name_truth"])) != normalize(str(row["company_name_ocr"]))
        changed_ocr_tokens = identify_changed_tokens(str(row["company_name_truth"]), str(row["company_name_ocr"]))
        changed_truth_tokens = identify_changed_tokens(str(row["company_name_ocr"]), str(row["company_name_truth"]))

        if corrupted:
            corrupted_confidences.append(confidence)
            if review_required:
                corrupted_flags += 1
            if weakest_token_matches_changed(weakest_token, changed_ocr_tokens):
                weakest_hits += 1
            if suggestion:
                suggestion_rows += 1
                if correction_exact_hit(changed_truth_tokens, suggestion):
                    suggestion_exact_hits += 1
                original_similarity = similarity(normalize(str(row["company_name_ocr"])), normalize(str(row["company_name_truth"])))
                corrected = apply_suggested_correction(str(row["company_name_ocr"]), weakest_token, suggestion)
                corrected_similarity = similarity(normalize(corrected), normalize(str(row["company_name_truth"])))
                delta = corrected_similarity - original_similarity
                similarity_deltas.append(delta)
                if corrected_similarity > original_similarity + 1e-9:
                    suggestion_improvements += 1
        else:
            clean_confidences.append(confidence)
            if review_required:
                clean_flags += 1

    confidence_separation = average(clean_confidences) - average(corrupted_confidences)
    clean_flag_rate = clean_flags / max(1, len(clean_confidences))
    corrupted_flag_rate = corrupted_flags / max(1, len(corrupted_confidences))
    weakest_token_hit_rate = weakest_hits / max(1, len(corrupted_confidences))
    suggestion_coverage_rate = suggestion_rows / max(1, len(corrupted_confidences))
    suggestion_exact_hit_rate = suggestion_exact_hits / max(1, suggestion_rows)
    suggestion_improvement_rate = suggestion_improvements / max(1, suggestion_rows)
    avg_similarity_delta = average(similarity_deltas)

    composite = compute_composite_score(
        confidence_separation=confidence_separation,
        clean_flag_rate=clean_flag_rate,
        corrupted_flag_rate=corrupted_flag_rate,
        weakest_token_hit_rate=weakest_token_hit_rate,
        suggestion_exact_hit_rate=suggestion_exact_hit_rate,
        suggestion_improvement_rate=suggestion_improvement_rate,
    )

    return OCRCompanyTierResult(
        tier=tier,
        confidence_separation=round(confidence_separation, 4),
        clean_flag_rate=round(clean_flag_rate, 4),
        corrupted_flag_rate=round(corrupted_flag_rate, 4),
        weakest_token_hit_rate=round(weakest_token_hit_rate, 4),
        suggestion_coverage_rate=round(suggestion_coverage_rate, 4),
        suggestion_exact_hit_rate=round(suggestion_exact_hit_rate, 4),
        suggestion_improvement_rate=round(suggestion_improvement_rate, 4),
        avg_similarity_delta_on_suggestions=round(avg_similarity_delta, 4),
        composite=round(composite, 4),
        rows=len(rows),
        time_seconds=round(time_seconds, 3),
        memory_mb=round(memory_mb, 1),
    )


def compute_composite_score(
    confidence_separation: float,
    clean_flag_rate: float,
    corrupted_flag_rate: float,
    weakest_token_hit_rate: float,
    suggestion_exact_hit_rate: float,
    suggestion_improvement_rate: float,
) -> float:
    separation_score = min(1.0, max(0.0, confidence_separation / 0.6))
    clean_score = 1.0 - clean_flag_rate
    return (
        0.25 * separation_score
        + 0.20 * clean_score
        + 0.20 * corrupted_flag_rate
        + 0.15 * weakest_token_hit_rate
        + 0.10 * suggestion_exact_hit_rate
        + 0.10 * suggestion_improvement_rate
    )


def identify_changed_tokens(left: str, right: str) -> list[str]:
    left_tokens = tokenize_for_alignment(left)
    right_tokens = tokenize_for_alignment(right)
    matcher = SequenceMatcher(None, left_tokens, right_tokens)
    changed: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed.extend(right_tokens[j1:j2])
    if not changed and normalize(left) != normalize(right):
        changed = right_tokens
    return dedupe_preserving_order(changed)


def tokenize_for_alignment(value: str) -> list[str]:
    return [token for token in normalize(value).replace(",", " ").replace(".", " ").split() if token]


def weakest_token_matches_changed(weakest_token: str, changed_tokens: list[str]) -> bool:
    if not weakest_token or not changed_tokens:
        return False
    normalized_weakest = normalize_token(weakest_token)
    for token in changed_tokens:
        normalized = normalize_token(token)
        if normalized_weakest == normalized or normalized_weakest in normalized or normalized in normalized_weakest:
            return True
        if similarity(normalized_weakest, normalized) >= 0.7:
            return True
    return False


def correction_exact_hit(changed_truth_tokens: list[str], suggestion: str) -> bool:
    if not suggestion:
        return False
    normalized_suggestion = normalize_token(suggestion)
    return any(normalize_token(token) == normalized_suggestion for token in changed_truth_tokens)


def apply_suggested_correction(value: str, weakest_token: str, suggestion: str) -> str:
    if not weakest_token or not suggestion:
        return value
    tokens = value.split()
    normalized_weakest = normalize_token(weakest_token)
    replacement = suggestion.split() if suggestion.split() else [suggestion]
    for index, token in enumerate(tokens):
        if normalize_token(token) == normalized_weakest:
            return " ".join(tokens[:index] + replacement + tokens[index + 1:])
    return value


def normalize(value: str) -> str:
    return " ".join(str(value).strip().upper().split())


def normalize_token(value: str) -> str:
    return "".join(character for character in normalize(value) if character.isalnum())


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_token(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
