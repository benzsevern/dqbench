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

        # Stage 1: Transform with GoldenFlow (configured, not zero-config)
        from goldenflow.config.schema import GoldenFlowConfig, TransformSpec

        df = pl.read_csv(csv_path)

        flow_config = GoldenFlowConfig(
            transforms=[
                TransformSpec(column="first_name", ops=["strip", "title_case"]),
                TransformSpec(column="last_name", ops=["strip", "title_case"]),
                TransformSpec(column="email", ops=["strip", "lowercase"]),
                TransformSpec(column="phone", ops=["strip", "phone_national"]),
                TransformSpec(column="address", ops=["strip", "collapse_whitespace"]),
                TransformSpec(column="city", ops=["strip", "title_case"]),
                TransformSpec(column="company", ops=["strip", "collapse_whitespace"]),
            ],
        )

        transform_result = goldenflow.transform_df(df, config=flow_config)
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

        # Build output: unique records + one representative per cluster.
        # For each cluster, pick the member with the lowest _row_id
        # (the original record, not the duplicate).
        internal_cols = ["__cluster_id__", "__golden_confidence__", "__row_id__", "__source__"]
        mk_pattern = "__mk_"

        def _drop_internal(df: pl.DataFrame) -> pl.DataFrame:
            drop = [c for c in df.columns if c in internal_cols or c.startswith(mk_pattern)]
            return df.drop(drop) if drop else df

        parts = []

        # Add unique (non-duplicate) records
        if dedupe_result.unique is not None and dedupe_result.unique.shape[0] > 0:
            parts.append(_drop_internal(dedupe_result.unique))

        # For clusters, pick the record with the lowest _row_id from each
        # cluster's members (original record, not duplicate)
        if dedupe_result.clusters and dedupe_result.dupes is not None:
            # Combine all records that are in clusters (dupes + golden)
            cluster_records = dedupe_result.dupes
            if dedupe_result.golden is not None:
                # Golden records also have the cluster data
                cluster_records = pl.concat([
                    cluster_records.select(dedupe_result.dupes.columns),
                    dedupe_result.golden.select(
                        [c for c in dedupe_result.dupes.columns if c in dedupe_result.golden.columns]
                    ),
                ], how="diagonal")

            if "_row_id" in cluster_records.columns:
                # For each cluster, keep the row with the lowest _row_id
                # Group by finding which cluster each record belongs to
                representatives = []
                for cluster in dedupe_result.clusters.values():
                    members = cluster["members"]
                    # Find the member rows in the cluster_records df
                    member_rows = cluster_records.filter(
                        pl.col("__row_id__").is_in(members)
                    )
                    if member_rows.shape[0] > 0 and "_row_id" in member_rows.columns:
                        # Pick the one with lowest _row_id
                        best = member_rows.sort("_row_id").head(1)
                        representatives.append(best)

                if representatives:
                    reps_df = pl.concat(representatives)
                    parts.append(_drop_internal(reps_df))

        if parts:
            # Align columns
            all_cols = parts[0].columns
            aligned = []
            for p in parts:
                missing = [c for c in all_cols if c not in p.columns]
                if missing:
                    p = p.with_columns([pl.lit(None).alias(c) for c in missing])
                aligned.append(p.select(all_cols))
            return pl.concat(aligned)

        return cleaned
