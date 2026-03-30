"""GoldenMatch adapter for DQBench ER benchmarks.

Uses GoldenMatch's full capabilities: standardization, multi-pass
blocking with phonetic/substring passes, ensemble scoring for names,
and proper cluster-based pair extraction.
"""
from __future__ import annotations
from pathlib import Path
from dqbench.adapters.base import EntityResolutionAdapter


class GoldenMatchAdapter(EntityResolutionAdapter):
    @property
    def name(self) -> str:
        return "goldenmatch"

    @property
    def version(self) -> str:
        try:
            import goldenmatch
            return goldenmatch.__version__
        except ImportError:
            return "not-installed"

    def deduplicate(self, csv_path: Path) -> list[tuple[int, int]]:
        try:
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
                "goldenmatch is not installed. Run: pip install goldenmatch"
            )

        import polars as pl
        df = pl.read_csv(csv_path)

        # Build a proper config using GoldenMatch's full capabilities.
        config = GoldenMatchConfig(
            # --- Standardize first: normalize data before comparison ---
            standardization=StandardizationConfig(
                email=["email"],
                phone=["phone"],
                first_name=["strip", "name_proper"],
                last_name=["strip", "name_proper"],
                address=["address"],
                zip=["zip5"] if "zip" in df.columns else [],
                state=["state"] if "state" in df.columns else [],
            ),

            # --- Multi-pass blocking: catch different dupe types ---
            # Pass 1: exact email (catches identical-email dupes)
            # Pass 2: soundex on last_name (catches phonetic variants)
            # Pass 3: first 3 chars of last_name (catches typos)
            blocking=BlockingConfig(
                strategy="multi_pass",
                keys=[
                    BlockingKeyConfig(
                        fields=["email"],
                        transforms=["lowercase", "strip"],
                    ),
                ],
                passes=[
                    BlockingKeyConfig(
                        fields=["email"],
                        transforms=["lowercase", "strip"],
                    ),
                    BlockingKeyConfig(
                        fields=["last_name"],
                        transforms=["soundex"],
                    ),
                    BlockingKeyConfig(
                        fields=["last_name"],
                        transforms=["substring:0:3"],
                    ),
                ],
            ),

            # --- Weighted matchkey: ensemble on names, exact on email ---
            matchkeys=[
                MatchkeyConfig(
                    name="identity",
                    type="weighted",
                    threshold=0.75,
                    fields=[
                        MatchkeyField(
                            field="first_name",
                            scorer="ensemble",
                            weight=1.0,
                            transforms=["lowercase", "strip"],
                        ),
                        MatchkeyField(
                            field="last_name",
                            scorer="ensemble",
                            weight=1.0,
                            transforms=["lowercase", "strip"],
                        ),
                        MatchkeyField(
                            field="email",
                            scorer="jaro_winkler",
                            weight=0.8,
                            transforms=["lowercase", "strip"],
                        ),
                        MatchkeyField(
                            field="phone",
                            scorer="exact",
                            weight=0.5,
                            transforms=["digits_only"],
                        ),
                        MatchkeyField(
                            field="address",
                            scorer="token_sort",
                            weight=0.6,
                            transforms=["lowercase", "strip"],
                        ),
                        MatchkeyField(
                            field="city",
                            scorer="exact",
                            weight=0.3,
                            transforms=["lowercase", "strip"],
                        ),
                    ],
                ),
            ],
        )

        # --- LLM scorer for borderline pairs (if API key available) ---
        import os
        has_llm = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if has_llm:
            config.llm_scorer = LLMScorerConfig(
                enabled=True,
                candidate_lo=0.60,   # send pairs in 0.60-0.90 range to LLM
                candidate_hi=0.90,
                auto_threshold=0.90, # auto-accept above 0.90
                budget=BudgetConfig(max_calls=500, max_cost_usd=1.0),
            )

        result = goldenmatch.dedupe_df(df, config=config)

        # Extract pairs from clusters
        pairs = []
        if result.clusters:
            for cluster in result.clusters.values():
                members = sorted(cluster["members"])
                if len(members) < 2:
                    continue
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        pairs.append((members[i], members[j]))

        return pairs
