"""DQBench CLI."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="dqbench", help="The standard benchmark for data quality tools.")


@app.command()
def run(
    adapter_name: str = typer.Argument(..., help="Adapter name (e.g., 'goldencheck') or --adapter path"),
    tier: Optional[int] = typer.Option(None, "--tier", "-t", help="Run specific tier only (1, 2, or 3)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    adapter_path: Optional[Path] = typer.Option(None, "--adapter", help="Path to custom adapter file"),
) -> None:
    """Run benchmark against a validation tool."""
    from dqbench.runner import run_benchmark

    adapter = _load_adapter(adapter_name, adapter_path)
    tiers = [tier] if tier else None
    scorecard = run_benchmark(adapter, tiers=tiers)

    if json_output:
        from dqbench.report import report_json

        report_json(scorecard, sys.stdout)
    else:
        from dqbench.report import report_rich

        report_rich(scorecard)


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
            if isinstance(obj, type) and hasattr(obj, "validate") and obj.__name__ != "DQBenchAdapter":
                return obj()
        raise typer.Exit("No DQBenchAdapter subclass found in adapter file.")

    builtin = {"goldencheck": "dqbench.adapters.goldencheck:GoldenCheckAdapter"}
    if name in builtin:
        module_path, class_name = builtin[name].split(":")
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)()

    raise typer.Exit(f"Unknown adapter: {name}. Use --adapter to specify a custom adapter file.")
