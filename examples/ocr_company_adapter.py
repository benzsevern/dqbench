"""Example OCR Company adapter for DQBench.

Usage:
    dqbench run placeholder --adapter examples/ocr_company_adapter.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from dqbench.adapters.base import OCRCompanyAdapter
from dqbench.models import OCRCompanyPrediction


class ExampleOCRCompanyAdapter(OCRCompanyAdapter):
    @property
    def name(self) -> str:
        return "ExampleOCRCompany"

    @property
    def version(self) -> str:
        return "0.1.0"

    def score_companies(self, csv_path: Path) -> list[OCRCompanyPrediction]:
        frame = pl.read_csv(csv_path)
        predictions: list[OCRCompanyPrediction] = []
        for row in frame.iter_rows(named=True):
            truth = str(row["company_name_truth"]).upper()
            ocr = str(row["company_name_ocr"]).upper()
            weakest_token = ""
            suggested = ""
            confidence = 0.9
            if truth != ocr:
                confidence = 0.25
                for ocr_token, truth_token in zip(ocr.split(), truth.split()):
                    if ocr_token != truth_token:
                        weakest_token = ocr_token
                        suggested = truth_token
                        break
            predictions.append(
                OCRCompanyPrediction(
                    record_id=str(row["record_id"]),
                    confidence=confidence,
                    weakest_token=weakest_token,
                    suggested_correction=suggested,
                )
            )
        return predictions
