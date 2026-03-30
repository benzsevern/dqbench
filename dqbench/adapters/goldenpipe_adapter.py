"""GoldenPipe adapter for DQBench Pipeline benchmarks.

Runs GoldenFlow transform then GoldenMatch dedupe with full capabilities:
standardization, multi-pass blocking, ensemble scoring, optional LLM.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
from dqbench.adapters.base import PipelineAdapter


class GoldenPipeAdapter(PipelineAdapter):
    @property
    def name(self) -> str:
        return "goldenpipe"

    @property
    def version(self) -> str:
        try:
            import goldenpipe
            return goldenpipe.__version__
        except ImportError:
            return "not-installed"

    def run_pipeline(self, csv_path: Path) -> pl.DataFrame:
        """Run the Golden Suite pipeline: transform then deduplicate."""
        try:
            import goldenflow
            import goldenmatch
            from goldenmatch.config.schemas import (
                GoldenMatchConfig,
                MatchkeyConfig,
                MatchkeyField,
                BlockingConfig,
                BlockingKeyConfig,
                StandardizationConfig,
                LLMScorerConfig,
                BudgetConfig,
            )
        except ImportError:
            raise RuntimeError(
                "goldenpipe[golden-suite] is not installed. "
                "Run: pip install goldenflow goldenmatch"
            )

        # Stage 1: Transform with GoldenFlow
        df = pl.read_csv(csv_path)
        transform_result = goldenflow.transform_df(df)
        cleaned = transform_result.df

        # Stage 2: Deduplicate with GoldenMatch (same config as ER adapter)
        config = GoldenMatchConfig(
            standardization=StandardizationConfig(
                email=["email"],
                phone=["phone"],
                first_name=["strip", "name_proper"],
                last_name=["strip", "name_proper"],
                address=["address"],
                zip=["zip5"] if "zip" in cleaned.columns else [],
                state=["state"] if "state" in cleaned.columns else [],
            ),
            blocking=BlockingConfig(
                strategy="multi_pass",
                keys=[
                    BlockingKeyConfig(fields=["email"], transforms=["lowercase", "strip"]),
                ],
                passes=[
                    BlockingKeyConfig(fields=["email"], transforms=["lowercase", "strip"]),
                    BlockingKeyConfig(fields=["last_name"], transforms=["soundex"]),
                    BlockingKeyConfig(fields=["last_name"], transforms=["substring:0:3"]),
                ],
            ),
            matchkeys=[
                MatchkeyConfig(
                    name="identity",
                    type="weighted",
                    threshold=0.75,
                    fields=[
                        MatchkeyField(field="first_name", scorer="ensemble", weight=1.0, transforms=["lowercase", "strip"]),
                        MatchkeyField(field="last_name", scorer="ensemble", weight=1.0, transforms=["lowercase", "strip"]),
                        MatchkeyField(field="email", scorer="jaro_winkler", weight=0.8, transforms=["lowercase", "strip"]),
                        MatchkeyField(field="phone", scorer="exact", weight=0.5, transforms=["digits_only"]),
                        MatchkeyField(field="address", scorer="token_sort", weight=0.6, transforms=["lowercase", "strip"]),
                        MatchkeyField(field="city", scorer="exact", weight=0.3, transforms=["lowercase", "strip"]),
                    ],
                ),
            ],
        )

        # Enable LLM scorer if API key available
        import os
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
            config.llm_scorer = LLMScorerConfig(
                enabled=True,
                candidate_lo=0.60,
                candidate_hi=0.90,
                auto_threshold=0.90,
                budget=BudgetConfig(max_calls=500, max_cost_usd=1.0),
            )

        dedupe_result = goldenmatch.dedupe_df(cleaned, config=config)

        # Build output: unique records + golden records (one per cluster)
        # Drop GoldenMatch internal columns to match expected schema
        internal_cols = [c for c in ["__cluster_id__", "__golden_confidence__", "__row_id__", "__source__"] if c != "_row_id"]

        parts = []
        if dedupe_result.unique is not None and dedupe_result.unique.shape[0] > 0:
            drop = [c for c in internal_cols if c in dedupe_result.unique.columns]
            # Also drop matchkey columns
            drop += [c for c in dedupe_result.unique.columns if c.startswith("__mk_")]
            parts.append(dedupe_result.unique.drop(drop))

        if dedupe_result.golden is not None and dedupe_result.golden.shape[0] > 0:
            drop = [c for c in internal_cols if c in dedupe_result.golden.columns]
            drop += [c for c in dedupe_result.golden.columns if c.startswith("__mk_")]
            parts.append(dedupe_result.golden.drop(drop))

        if parts:
            # Align columns across parts
            all_cols = parts[0].columns
            aligned = []
            for p in parts:
                for col in all_cols:
                    if col not in p.columns:
                        p = p.with_columns(pl.lit(None).alias(col))
                aligned.append(p.select(all_cols))
            return pl.concat(aligned)

        return cleaned
