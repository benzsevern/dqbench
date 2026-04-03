"""Tests for Pipeline tier generators."""
from __future__ import annotations
import polars as pl

from dqbench.generator.pipeline_tier1 import generate_pipeline_tier1
from dqbench.generator.pipeline_tier2 import generate_pipeline_tier2
from dqbench.generator.pipeline_tier3 import generate_pipeline_tier3
from dqbench.pipeline_ground_truth import PipelineGroundTruth


class TestPipelineTier1:
    def test_returns_correct_types(self):
        messy, clean, gt = generate_pipeline_tier1()
        assert isinstance(messy, pl.DataFrame)
        assert isinstance(clean, pl.DataFrame)
        assert isinstance(gt, PipelineGroundTruth)

    def test_row_counts(self):
        messy, clean, gt = generate_pipeline_tier1()
        assert messy.shape[0] == 1000
        assert clean.shape[0] == 900
        assert gt.rows == 1000
        assert gt.expected_output_rows == 900

    def test_has_row_id_column(self):
        messy, clean, gt = generate_pipeline_tier1()
        assert "_row_id" in messy.columns
        assert "_row_id" in clean.columns

    def test_duplicate_pair_count(self):
        _, _, gt = generate_pipeline_tier1()
        assert len(gt.duplicate_pairs) == 100

    def test_planted_issues_count(self):
        _, _, gt = generate_pipeline_tier1()
        assert gt.planted_issues == 150

    def test_determinism(self):
        m1, c1, gt1 = generate_pipeline_tier1()
        m2, c2, gt2 = generate_pipeline_tier1()
        assert m1.equals(m2)
        assert c1.equals(c2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, _, gt = generate_pipeline_tier1()
        assert gt.tier == 1
        assert gt.version == "1.0.0"


class TestPipelineTier2:
    def test_returns_correct_types(self):
        messy, clean, gt = generate_pipeline_tier2()
        assert isinstance(messy, pl.DataFrame)
        assert isinstance(clean, pl.DataFrame)
        assert isinstance(gt, PipelineGroundTruth)

    def test_row_counts(self):
        messy, clean, gt = generate_pipeline_tier2()
        assert messy.shape[0] == 5000
        assert clean.shape[0] == 4250
        assert gt.rows == 5000
        assert gt.expected_output_rows == 4250

    def test_has_row_id_column(self):
        messy, clean, _ = generate_pipeline_tier2()
        assert "_row_id" in messy.columns
        assert "_row_id" in clean.columns

    def test_duplicate_pair_count(self):
        _, _, gt = generate_pipeline_tier2()
        assert len(gt.duplicate_pairs) == 750

    def test_determinism(self):
        m1, c1, gt1 = generate_pipeline_tier2()
        m2, c2, gt2 = generate_pipeline_tier2()
        assert m1.equals(m2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, _, gt = generate_pipeline_tier2()
        assert gt.tier == 2


class TestPipelineTier3:
    def test_returns_correct_types(self):
        messy, clean, gt = generate_pipeline_tier3()
        assert isinstance(messy, pl.DataFrame)
        assert isinstance(clean, pl.DataFrame)
        assert isinstance(gt, PipelineGroundTruth)

    def test_row_counts(self):
        messy, clean, gt = generate_pipeline_tier3()
        assert messy.shape[0] == 10000
        assert clean.shape[0] == 8000
        assert gt.rows == 10000
        assert gt.expected_output_rows == 8000

    def test_has_row_id_column(self):
        messy, clean, _ = generate_pipeline_tier3()
        assert "_row_id" in messy.columns
        assert "_row_id" in clean.columns

    def test_duplicate_pair_count(self):
        _, _, gt = generate_pipeline_tier3()
        assert len(gt.duplicate_pairs) == 2000

    def test_determinism(self):
        m1, c1, gt1 = generate_pipeline_tier3()
        m2, c2, gt2 = generate_pipeline_tier3()
        assert m1.equals(m2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, _, gt = generate_pipeline_tier3()
        assert gt.tier == 3
