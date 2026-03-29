"""Tests for ER ground truth model and loader."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

from dqbench.er_ground_truth import ERGroundTruth, load_er_ground_truth


def test_er_ground_truth_creation():
    gt = ERGroundTruth(
        tier=1,
        version="1.0.0",
        rows=1000,
        duplicate_pairs=[(0, 900), (1, 901), (2, 902)],
        total_duplicates=3,
        difficulty="easy",
    )
    assert gt.tier == 1
    assert gt.rows == 1000
    assert len(gt.duplicate_pairs) == 3
    assert gt.difficulty == "easy"


def test_er_ground_truth_roundtrip():
    """Serialize to JSON and deserialize back — values must match."""
    gt = ERGroundTruth(
        tier=2,
        version="1.0.0",
        rows=5000,
        duplicate_pairs=[(10, 4300), (55, 4400)],
        total_duplicates=2,
        difficulty="fuzzy",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt.model_dump(), f)
        path = Path(f.name)

    loaded = load_er_ground_truth(path)
    assert loaded.tier == gt.tier
    assert loaded.version == gt.version
    assert loaded.rows == gt.rows
    assert loaded.duplicate_pairs == gt.duplicate_pairs
    assert loaded.total_duplicates == gt.total_duplicates
    assert loaded.difficulty == gt.difficulty
    path.unlink()


def test_er_ground_truth_pairs_are_tuples():
    """Pairs stored as lists in JSON should be converted to tuples."""
    data = {
        "tier": 1,
        "version": "1.0.0",
        "rows": 100,
        "duplicate_pairs": [[0, 50], [1, 51]],
        "total_duplicates": 2,
        "difficulty": "easy",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    loaded = load_er_ground_truth(path)
    for pair in loaded.duplicate_pairs:
        assert isinstance(pair, tuple)
    path.unlink()
