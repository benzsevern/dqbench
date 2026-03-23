"""GoldenCheck adapter for DQBench."""
from __future__ import annotations
from pathlib import Path

from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding


class GoldenCheckAdapter(DQBenchAdapter):
    @property
    def name(self) -> str:
        return "GoldenCheck"

    @property
    def version(self) -> str:
        try:
            from goldencheck import __version__

            return __version__
        except ImportError:
            return "not installed"

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        from goldencheck.engine.scanner import scan_file

        findings, _ = scan_file(csv_path)
        return [
            DQBenchFinding(
                column=f.column,
                severity=f.severity.name.lower(),
                check=f.check,
                message=f.message,
                confidence=getattr(f, "confidence", 1.0),
            )
            for f in findings
        ]
