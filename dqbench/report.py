"""Rich console and JSON report for DQBench scorecards."""
from __future__ import annotations
import dataclasses
import json
from typing import IO

from rich.console import Console
from rich.table import Table
from rich import box

from dqbench.models import Scorecard


def report_rich(scorecard: Scorecard) -> None:
    """Print a Rich formatted scorecard to the console."""
    console = Console()

    console.print()
    console.print("[bold cyan]DQBench Report[/bold cyan]", justify="center")
    console.print(f"[bold]Tool:[/bold] {scorecard.tool_name}  [bold]Version:[/bold] {scorecard.tool_version}")
    console.print()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Tier", style="bold", justify="center")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("FPR", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Findings", justify="right")

    for t in scorecard.tiers:
        table.add_row(
            str(t.tier),
            f"{t.recall:.1%}",
            f"{t.precision:.1%}",
            f"{t.f1:.1%}",
            f"{t.false_positive_rate:.1%}",
            f"{t.time_seconds:.3f}",
            f"{t.memory_mb:.1f}",
            str(t.findings_count),
        )

    console.print(table)
    console.print()

    # DQBench Score breakdown
    weights = {1: 0.20, 2: 0.40, 3: 0.40}
    parts = []
    for t in scorecard.tiers:
        w = weights.get(t.tier, 0)
        parts.append(f"T{t.tier}: {t.f1:.1%} × {int(w * 100)}%")

    score_breakdown = "  +  ".join(parts)
    console.print(f"[bold]DQBench Score:[/bold] [bold green]{scorecard.dqbench_score:.2f}[/bold green]  ({score_breakdown})")

    if scorecard.llm_cost is not None:
        console.print(f"[bold]LLM Cost:[/bold] ${scorecard.llm_cost:.4f}")

    if scorecard.confidence_calibration is not None:
        console.print(f"[bold]Confidence Calibration:[/bold] {scorecard.confidence_calibration:.4f}")

    console.print()


def report_json(scorecard: Scorecard, output: IO[str]) -> None:
    """Serialize the scorecard to JSON and write to output stream."""
    data = {
        "tool_name": scorecard.tool_name,
        "tool_version": scorecard.tool_version,
        "dqbench_score": scorecard.dqbench_score,
        "tiers": [dataclasses.asdict(t) for t in scorecard.tiers],
    }
    if scorecard.llm_cost is not None:
        data["llm_cost"] = scorecard.llm_cost
    if scorecard.confidence_calibration is not None:
        data["confidence_calibration"] = scorecard.confidence_calibration
    json.dump(data, output, indent=2)
    output.write("\n")
