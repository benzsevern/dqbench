"""Rich console and JSON report for DQBench scorecards."""
from __future__ import annotations
import dataclasses
import json
from typing import IO

from rich.console import Console
from rich.table import Table
from rich import box

from dqbench.models import Scorecard, TransformScorecard


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


def report_comparison(scorecards: list[Scorecard]) -> None:
    """Print a head-to-head comparison table for multiple scorecards."""
    console = Console()

    # Determine which tiers are present
    all_tiers: list[int] = []
    for sc in scorecards:
        for t in sc.tiers:
            if t.tier not in all_tiers:
                all_tiers.append(t.tier)
    all_tiers = sorted(all_tiers)

    console.print()
    console.rule("[bold cyan]DQBench v1.0 — Head-to-Head Comparison[/bold cyan]")
    console.print()

    # Build table
    table = Table(
        title="Issue-Level F1 by Tier  (DQBench Score = weighted average)",
        box=box.HEAVY_HEAD,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="cyan",
        show_lines=True,
    )

    table.add_column("Tool", style="bold", min_width=28, no_wrap=True)
    for tier in all_tiers:
        table.add_column(f"T{tier} F1", justify="right", min_width=8)
    table.add_column("Score", justify="right", style="bold green", min_width=7)
    table.add_column("Findings", justify="right", min_width=9)

    for sc in scorecards:
        tier_map = {t.tier: t for t in sc.tiers}
        row = [sc.tool_name]
        total_findings = 0
        for tier in all_tiers:
            t = tier_map.get(tier)
            if t:
                row.append(f"{t.issue_f1:.1%}")
                total_findings += t.findings_count
            else:
                row.append("—")
        row.append(f"{sc.dqbench_score:.2f}")
        row.append(str(total_findings))
        table.add_row(*row)

    console.print(table)
    console.print()

    # Highlight winner
    if scorecards:
        best = max(scorecards, key=lambda s: s.dqbench_score)
        console.print(
            f"[bold]Winner:[/bold] [bold green]{best.tool_name}[/bold green]  "
            f"[dim]DQBench Score = {best.dqbench_score:.2f}[/dim]"
        )
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


def report_transform_rich(scorecard: TransformScorecard) -> None:
    """Pretty-print transform benchmark results."""
    console = Console()
    console.print(
        f"\n[bold]Transform Benchmark: {scorecard.tool_name} v{scorecard.tool_version}[/bold]\n"
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Tier", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Correct", style="green")
    table.add_column("Wrong", style="red")
    table.add_column("Skipped", style="yellow")
    table.add_column("Planted", style="dim")
    table.add_column("Time", style="dim")
    table.add_column("Memory", style="dim")

    for t in scorecard.tiers:
        table.add_row(
            f"T{t.tier}",
            f"{t.accuracy:.1%}",
            str(t.correct_cells),
            str(t.wrong_cells),
            str(t.skipped_cells),
            str(t.planted_cells),
            f"{t.time_seconds:.2f}s",
            f"{t.memory_mb:.1f} MB",
        )

    console.print(table)
    console.print(
        f"\n[bold]DQBench Transform Score: {scorecard.composite_score:.2f} / 100[/bold]\n"
    )

    # Per-column breakdown for each tier
    for t in scorecard.tiers:
        if not t.per_column:
            continue
        col_table = Table(
            title=f"Tier {t.tier}: Per-Column Accuracy",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
        )
        col_table.add_column("Column", style="cyan")
        col_table.add_column("Accuracy", style="green")
        col_table.add_column("Correct", style="green")
        col_table.add_column("Planted", style="dim")

        for c in sorted(t.per_column, key=lambda x: x.accuracy):
            col_table.add_row(
                c.column,
                f"{c.accuracy:.1%}",
                str(c.correct_cells),
                str(c.planted_cells),
            )

        console.print(col_table)


def report_transform_json(scorecard: TransformScorecard, output: IO[str]) -> None:
    """Serialize a TransformScorecard to JSON and write to output stream."""
    data = {
        "tool_name": scorecard.tool_name,
        "tool_version": scorecard.tool_version,
        "composite_score": scorecard.composite_score,
        "tiers": [dataclasses.asdict(t) for t in scorecard.tiers],
    }
    json.dump(data, output, indent=2)
    output.write("\n")
