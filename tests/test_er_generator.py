"""Tests for ER tier generators."""
from __future__ import annotations
import polars as pl
import pytest

from dqbench.generator.er_tier1 import generate_er_tier1
from dqbench.er_ground_truth import ERGroundTruth


class TestERTier1:
    def test_returns_dataframe_and_ground_truth(self):
        df, gt = generate_er_tier1()
        assert isinstance(df, pl.DataFrame)
        assert isinstance(gt, ERGroundTruth)

    def test_row_count(self):
        df, gt = generate_er_tier1()
        assert df.shape[0] == 1000
        assert gt.rows == 1000

    def test_expected_columns(self):
        df, _ = generate_er_tier1()
        expected = {"first_name", "last_name", "email", "phone",
                    "address", "city", "state", "zip", "company"}
        assert set(df.columns) == expected

    def test_duplicate_pair_count(self):
        _, gt = generate_er_tier1()
        assert gt.total_duplicates == 100
        assert len(gt.duplicate_pairs) == 100

    def test_duplicate_pair_breakdown(self):
        """50 case-change, 30 typo, 20 name-swap = 100 total."""
        _, gt = generate_er_tier1()
        assert len(gt.duplicate_pairs) == 100

    def test_pairs_reference_valid_rows(self):
        df, gt = generate_er_tier1()
        n = df.shape[0]
        for a, b in gt.duplicate_pairs:
            assert 0 <= a < n, f"Invalid row index {a}"
            assert 0 <= b < n, f"Invalid row index {b}"
            assert a != b, f"Self-pair ({a}, {a})"

    def test_determinism(self):
        """Two calls produce identical output."""
        df1, gt1 = generate_er_tier1()
        df2, gt2 = generate_er_tier1()
        assert df1.equals(df2)
        assert gt1.duplicate_pairs == gt2.duplicate_pairs

    def test_ground_truth_metadata(self):
        _, gt = generate_er_tier1()
        assert gt.tier == 1
        assert gt.difficulty == "easy"
        assert gt.version == "1.0.0"
