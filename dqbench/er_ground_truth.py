"""Load and query ER ground truth manifests."""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel


class ERGroundTruth(BaseModel):
    tier: int
    version: str
    rows: int
    duplicate_pairs: list[tuple[int, int]]
    total_duplicates: int
    difficulty: str  # "easy", "fuzzy", "adversarial"


def load_er_ground_truth(path: Path) -> ERGroundTruth:
    with open(path) as f:
        return ERGroundTruth(**json.load(f))
