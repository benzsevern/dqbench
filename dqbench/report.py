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

    # Column-level table
    col_table = Table(
        title="Column-Level Detection",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    col_table.add_column("Tier", style="bold", justify="center")
    col_table.add_column("Recall", justify="right")
    col_table.add_column("Precision", justify="right")
    col_table.add_column("F1", justify="right")
    col_table.add_column("FPR", justify="right")
    col_table.add_column("Time (s)", justify="right")
    col_table.add_column("Memory (MB)", justify="right")
    col_table.add_column("Findings", justify="right")

    for t in scorecard.tiers:
        col_table.add_row(
            str(t.tier),
            f"{t.recall:.1%}",
            f"{t.precision:.1%}",
            f"{t.f1:.1%}",
            f"{t.false_positive_rate:.1%}",
            f"{t.time_seconds:.3f}",
            f"{t.memory_mb:.1f}",
            str(t.findings_count),
        )

    console.print(col_table)
    console.print()

    # Issue-level table
    issue_table = Table(
        title="Issue-Level Detection (Targeted)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold yellow",
    )
    issue_table.add_column("Tier", style="bold", justify="center")
    issue_table.add_column("Issue Recall", justify="right")
    issue_table.add_column("Issue Precision", justify="right")
    issue_table.add_column("Issue F1", justify="right")

    for t in scorecard.tiers:
        issue_table.add_row(
            str(t.tier),
            f"{t.issue_recall:.1%}",
            f"{t.issue_precision:.1%}",
            f"{t.issue_f1:.1%}",
        )

    console.print(issue_table)
    console.print()

    # DQBench Score breakdown (now based on issue_f1)
    weights = {1: 0.20, 2: 0.40, 3: 0.40}
    parts = []
    for t in scorecard.tiers:
        w = weights.get(t.tier, 0)
        parts.append(f"T{t.tier}: {t.issue_f1:.1%} x {int(w * 100)}%")

    score_breakdown = "  +  ".join(parts)
    console.print(
        f"[bold]DQBench Score:[/bold] [bold green]{scorecard.dqbench_score:.2f}[/bold green]"
        f"  ({score_breakdown})  [dim](issue-level F1)[/dim]"
    )

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
