import polars as pl
from dqbench.transform_scorer import score_transform_tier


def test_perfect_transform():
    clean = pl.DataFrame({"a": ["correct", "correct"], "b": ["fixed", "fixed"]})
    messy = pl.DataFrame({"a": ["wrong", "wrong"], "b": ["broken", "broken"]})
    result = clean.clone()  # perfect output

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.accuracy == 1.0
    assert tier.correct_cells == 4
    assert tier.wrong_cells == 0


def test_no_op_transform():
    clean = pl.DataFrame({"a": ["correct"], "b": ["fixed"]})
    messy = pl.DataFrame({"a": ["wrong"], "b": ["broken"]})
    result = messy.clone()  # tool did nothing

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.accuracy == 0.0
    assert tier.skipped_cells == 2


def test_partial_transform():
    clean = pl.DataFrame({"a": ["correct", "correct"], "b": ["same", "same"]})
    messy = pl.DataFrame({"a": ["wrong", "correct"], "b": ["same", "same"]})
    result = pl.DataFrame({"a": ["correct", "correct"], "b": ["same", "same"]})

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.accuracy == 1.0  # only 1 planted cell, and it's correct
    assert tier.planted_cells == 1


def test_wrong_shape():
    clean = pl.DataFrame({"a": ["correct"]})
    messy = pl.DataFrame({"a": ["wrong"]})
    result = pl.DataFrame({"a": ["correct", "extra"]})  # wrong row count

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.accuracy == 0.0


def test_column_mismatch():
    clean = pl.DataFrame({"a": ["correct"], "b": ["fixed"]})
    messy = pl.DataFrame({"a": ["wrong"], "b": ["broken"]})
    result = pl.DataFrame({"a": ["correct"], "c": ["different_col"]})  # wrong column

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.accuracy == 0.0


def test_no_planted_issues():
    """All cells same in clean and messy → nothing to score → accuracy 0.0."""
    clean = pl.DataFrame({"a": ["same", "same"]})
    messy = pl.DataFrame({"a": ["same", "same"]})
    result = pl.DataFrame({"a": ["same", "same"]})

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.planted_cells == 0
    assert tier.accuracy == 0.0


def test_per_column_results():
    clean = pl.DataFrame({"phone": ["(555) 123-4567", "(555) 234-5678"], "status": ["active", "inactive"]})
    messy = pl.DataFrame({"phone": ["555-123-4567", "5552345678"], "status": ["actve", "inactive"]})
    result = pl.DataFrame({"phone": ["(555) 123-4567", "(555) 234-5678"], "status": ["active", "inactive"]})

    tier = score_transform_tier(result, clean, messy, tier=1, time_seconds=0.1, memory_mb=1.0)
    assert tier.accuracy == 1.0
    col_names = {c.column for c in tier.per_column}
    # status row 1 is same in messy and clean → only phone and status[0] are planted
    assert "phone" in col_names
