"""Tests for Pipeline scorer."""
from __future__ import annotations

import polars as pl
import pytest

from dqbench.pipeline_scorer import score_pipeline_tier
from dqbench.models import PipelineTierResult


class TestScorePipelineTier:
    @pytest.fixture
    def tier_dir(self, tmp_path):
        """Create a tier directory with clean ground truth CSV."""
        clean_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        messy_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4", "5"],
            "name": ["alice", "BOB", "Charlie", "diana", "Eve", "Alice"],
            "email": ["a@x.com", "b@x.com", "C@X.COM", "d@x.com", "e@x.com", "a@x.com"],
        })
        clean_df.write_csv(tmp_path / "data_clean_deduped.csv")
        messy_df.write_csv(tmp_path / "data.csv")
        return tmp_path

    def test_perfect_output(self, tier_dir):
        """Output matches clean ground truth exactly."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        assert isinstance(result, PipelineTierResult)
        assert result.transform_accuracy == 1.0
        assert result.dedup_accuracy == 1.0
        assert result.composite == 1.0

    def test_wrong_row_count(self, tier_dir):
        """Output has too many rows -- dedup_accuracy penalized."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4", "5"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Alice"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "a@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        # dedup_accuracy = 1.0 - abs(6 - 5) / 5 = 0.8
        assert result.dedup_accuracy == pytest.approx(0.8)
        assert result.output_rows == 6
        assert result.expected_rows == 5

    def test_wrong_cell_values(self, tier_dir):
        """Some cells not cleaned -- transform_accuracy penalized."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["alice", "Bob", "Charlie", "Diana", "Eve"],  # "alice" not fixed
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        assert result.transform_accuracy < 1.0
        assert result.dedup_accuracy == 1.0

    def test_empty_output(self, tier_dir):
        """Empty DataFrame -- worst possible scores."""
        result_df = pl.DataFrame({
            "_row_id": pl.Series([], dtype=pl.Utf8),
            "name": pl.Series([], dtype=pl.Utf8),
            "email": pl.Series([], dtype=pl.Utf8),
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        assert result.transform_accuracy == 0.0
        assert result.dedup_accuracy == 0.0
        assert result.composite == 0.0

    def test_composite_calculation(self, tier_dir):
        """Composite = transform_accuracy * 0.6 + dedup_accuracy * 0.4."""
        result_df = pl.DataFrame({
            "_row_id": ["0", "1", "2", "3", "4"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        })
        result = score_pipeline_tier(
            result_df, tier_dir, tier=1,
            time_seconds=1.0, memory_mb=10.0,
            expected_rows=5,
        )
        expected_composite = result.transform_accuracy * 0.6 + result.dedup_accuracy * 0.4
        assert result.composite == pytest.approx(expected_composite)
