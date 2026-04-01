"""Tests for the DQBench CLI."""
from __future__ import annotations

from typer.testing import CliRunner

from dqbench.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dqbench" in result.stdout.lower()


def test_generate():
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0


def test_results_command():
    result = runner.invoke(app, ["results"])
    assert result.exit_code == 0
    assert "dqbench run" in result.stdout


def test_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "adapter" in result.stdout.lower()


def test_generate_help():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "force" in result.stdout.lower()


def test_generate_ocr_company():
    result = runner.invoke(app, ["generate", "--ocr-company"])
    assert result.exit_code == 0
