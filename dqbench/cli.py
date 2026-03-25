"""DQBench CLI."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="dqbench", help="The standard benchmark for data quality tools.")

# ---------------------------------------------------------------------------
# Built-in adapter registry
# ---------------------------------------------------------------------------
BUILTIN_ADAPTERS: dict[str, str] = {
    # GoldenCheck
    "goldencheck": "dqbench.adapters.goldencheck:GoldenCheckAdapter",
    # GoldenFlow
    "goldenflow": "dqbench.adapters.goldenflow:GoldenFlowAdapter",
    # Great Expectations
    "gx-zero":     "dqbench.adapters.great_expectations_adapter:GXZeroConfigAdapter",
    "gx-auto":     "dqbench.adapters.great_expectations_adapter:GXAutoProfileAdapter",
    "gx-best":     "dqbench.adapters.great_expectations_adapter:GXBestEffortAdapter",
    # Pandera
    "pandera-zero": "dqbench.adapters.pandera_adapter:PanderaZeroConfigAdapter",
    "pandera-auto": "dqbench.adapters.pandera_adapter:PanderaAutoProfileAdapter",
    "pandera-best": "dqbench.adapters.pandera_adapter:PanderaBestEffortAdapter",
    # Soda
    "soda-zero":   "dqbench.adapters.soda_adapter:SodaZeroConfigAdapter",
    "soda-auto":   "dqbench.adapters.soda_adapter:SodaAutoProfileAdapter",
    "soda-best":   "dqbench.adapters.soda_adapter:SodaBestEffortAdapter",
}

# Order for the comparison table
ALL_ADAPTER_NAMES = [
    "goldencheck",
    "goldenflow",
    "gx-zero",
    "gx-auto",
    "gx-best",
    "pandera-zero",
    "pandera-auto",
    "pandera-best",
    "soda-zero",
    "soda-auto",
    "soda-best",
]


@app.command()
def run(
    adapter_name: str = typer.Argument(..., help=(
        "Adapter name: goldencheck | gx-zero | gx-auto | gx-best | "
        "pandera-zero | pandera-auto | pandera-best | "
        "soda-zero | soda-auto | soda-best | all"
    )),
    tier: Optional[int] = typer.Option(None, "--tier", "-t", help="Run specific tier only (1, 2, or 3)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    adapter_path: Optional[Path] = typer.Option(None, "--adapter", help="Path to custom adapter file"),
) -> None:
    """Run benchmark against a validation tool."""
    from dqbench.runner import run_benchmark

    if adapter_name == "all":
        _run_all(tier=tier)
        return

    adapter = _load_adapter(adapter_name, adapter_path)
    tiers = [tier] if tier else None

    from dqbench.adapters.base import TransformAdapter
    if isinstance(adapter, TransformAdapter):
        from dqbench.runner import run_transform_benchmark
        from dqbench.report import report_transform_rich, report_transform_json
        scorecard = run_transform_benchmark(adapter, tiers=tiers)
        if json_output:
            report_transform_json(scorecard, sys.stdout)
        else:
            report_transform_rich(scorecard)
    else:
        scorecard = run_benchmark(adapter, tiers=tiers)
        if json_output:
            from dqbench.report import report_json
            report_json(scorecard, sys.stdout)
        else:
            from dqbench.report import report_rich
            report_rich(scorecard)


def _run_all(tier: Optional[int] = None) -> None:
    """Run all registered adapters and print a comparison table."""
    from dqbench.runner import run_benchmark
    from dqbench.report import report_comparison

    scorecards = []
    tiers = [tier] if tier else None

    for name in ALL_ADAPTER_NAMES:
        typer.echo(f"\nRunning: {name} ...", err=True)
        try:
            adapter = _load_adapter(name)
            sc = run_benchmark(adapter, tiers=tiers)
            scorecards.append(sc)
            typer.echo(f"  Done — score: {sc.dqbench_score:.2f}", err=True)
        except Exception as e:
            typer.echo(f"  FAILED: {e}", err=True)

    report_comparison(scorecards)


@app.command()
def generate(
    force: bool = typer.Option(False, "--force", help="Regenerate even if cached"),
) -> None:
    """Generate benchmark datasets."""
    from dqbench.runner import CACHE_DIR, ensure_datasets

    if force:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
    ensure_datasets()
    typer.echo(f"Datasets generated at {CACHE_DIR}")


@app.command()
def results() -> None:
    """Show results from last run."""
    typer.echo("No cached results. Run 'dqbench run <adapter>' first.")


def _load_adapter(name: str, path: Path | None = None):
    """Load a built-in or custom adapter by name or file path."""
    if path:
        import importlib.util

        spec = importlib.util.spec_from_file_location("custom_adapter", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and (hasattr(obj, "validate") or hasattr(obj, "transform")) and obj.__name__ not in ("DQBenchAdapter", "TransformAdapter"):
                return obj()
        raise typer.Exit("No DQBenchAdapter subclass found in adapter file.")

    if name in BUILTIN_ADAPTERS:
        module_path, class_name = BUILTIN_ADAPTERS[name].split(":")
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)()

    raise typer.Exit(
        f"Unknown adapter: '{name}'. "
        f"Available: {', '.join(ALL_ADAPTER_NAMES + ['all'])} or use --adapter for a custom file."
    )
