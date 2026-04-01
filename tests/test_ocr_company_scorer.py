from __future__ import annotations

import polars as pl

from dqbench.models import OCRCompanyPrediction
from dqbench.ocr_company_scorer import score_ocr_company_tier


def test_ocr_company_scorer_rewards_good_predictions():
    dataset = pl.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "document_type": ["invoice"] * 4,
            "company_name_truth": ["ATLAS LLC", "BLUE RIDGE INC", "SUMMIT LLC", "RIVER LLC"],
            "company_name_ocr": ["ATLAS LLC", "BLUE R1DGE INC", "SUMMIT ILC", "RIVER LLC"],
            "company_corrupted": [False, True, True, False],
        }
    )
    predictions = [
        OCRCompanyPrediction(record_id="a", confidence=0.95),
        OCRCompanyPrediction(record_id="b", confidence=0.15, weakest_token="R1DGE", suggested_correction="RIDGE"),
        OCRCompanyPrediction(record_id="c", confidence=0.10, weakest_token="ILC", suggested_correction="LLC"),
        OCRCompanyPrediction(record_id="d", confidence=0.91),
    ]
    result = score_ocr_company_tier(predictions, dataset, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert result.corrupted_flag_rate == 1.0
    assert result.clean_flag_rate == 0.0
    assert result.suggestion_exact_hit_rate == 1.0
    assert result.suggestion_improvement_rate == 1.0
    assert result.composite > 0.8
