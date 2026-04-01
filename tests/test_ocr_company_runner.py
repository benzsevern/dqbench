"""Tests for OCR company benchmark runner."""
from __future__ import annotations

from pathlib import Path

from dqbench.adapters.base import OCRCompanyAdapter
from dqbench.models import OCRCompanyPrediction
from dqbench.runner import run_ocr_company_benchmark


class NullOCRCompanyAdapter(OCRCompanyAdapter):
    @property
    def name(self) -> str:
        return "NullOCR"

    @property
    def version(self) -> str:
        return "0.0"

    def score_companies(self, csv_path: Path) -> list[OCRCompanyPrediction]:
        return []


def test_run_ocr_company_benchmark_returns_scorecard():
    scorecard = run_ocr_company_benchmark(NullOCRCompanyAdapter(), tiers=[1])
    assert scorecard.tool_name == "NullOCR"
    assert len(scorecard.tiers) == 1
    assert scorecard.tiers[0].tier == 1
    assert scorecard.tiers[0].corrupted_flag_rate == 0.0
