"""Load and query ground truth manifests."""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel


class PlantedColumn(BaseModel):
    issues: list[str]
    planted_count: int
    description: str
    affected_rows: list[int] = []


class GroundTruth(BaseModel):
    tier: int
    version: str
    rows: int
    columns: int
    planted_columns: dict[str, PlantedColumn]
    clean_columns: list[str]
    total_planted_issues: int


def load_ground_truth(path: Path) -> GroundTruth:
    with open(path) as f:
        return GroundTruth(**json.load(f))
