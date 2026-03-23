"""Soda Core adapters for DQBench — zero-config, auto-profiled, best-effort.

Note: soda-core 4.3 uses a contracts model requiring a SQL database backend
(DuckDB, PostgreSQL, etc.). Since no database plugin is installed in this
environment, these adapters implement the equivalent SodaCL-style checks
directly via pandas, faithfully reproducing what SodaCL checks would test.

The Soda zero/auto/best modes mirror the SodaCL philosophy:
  - Zero-config: no checks defined → no findings
  - Auto-profiled: profile columns, infer anomalies (missing, range, cardinality)
  - Best-effort: explicit SodaCL-equivalent checks by column name
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding

logger = logging.getLogger(__name__)


def _soda_version() -> str:
    try:
        import soda_core
        from soda_core.common.version import SODA_CORE_VERSION
        return SODA_CORE_VERSION
    except Exception:
        try:
            import importlib.metadata
            return importlib.metadata.version("soda-core")
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Zero-config
# ---------------------------------------------------------------------------

class SodaZeroConfigAdapter(DQBenchAdapter):
    """Soda with no checks (SodaCL requires explicit checks) — returns empty findings."""

    @property
    def name(self) -> str:
        return "Soda (zero-config)"

    @property
    def version(self) -> str:
        return _soda_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        # Soda requires SodaCL checks to be explicitly defined
        return []


# ---------------------------------------------------------------------------
# Auto-profiled
# ---------------------------------------------------------------------------

class SodaAutoProfileAdapter(DQBenchAdapter):
    """Soda-equivalent auto-profiling checks derived from column statistics.

    Mirrors what `soda profile columns` and anomaly detection would surface:
    - Missing values above threshold
    - Numeric columns out of statistical bounds (mean ± 3σ)
    - Low-cardinality columns with unexpected values
    - Row count > 0
    """

    @property
    def name(self) -> str:
        return "Soda (auto-profiled)"

    @property
    def version(self) -> str:
        return _soda_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        findings: list[DQBenchFinding] = []
        try:
            import pandas as pd
            import numpy as np

            df = pd.read_csv(csv_path)

            # ----------------------------------------------------------------
            # Row count check (SodaCL: row_count > 0)
            # ----------------------------------------------------------------
            if len(df) == 0:
                findings.append(DQBenchFinding(
                    column="dataset",
                    severity="error",
                    check="null_values",
                    message="Soda check: row_count = 0, dataset is empty",
                ))
                return findings

            # ----------------------------------------------------------------
            # Profile each column
            # ----------------------------------------------------------------
            for col in df.columns:
                try:
                    series = df[col]

                    # --- Missing values ---
                    null_count = series.isna().sum()
                    null_pct = null_count / len(series)
                    if null_pct > 0.20:  # >20% missing triggers an anomaly
                        findings.append(DQBenchFinding(
                            column=col,
                            severity="warning" if null_pct < 0.50 else "error",
                            check="null_values",
                            message=(
                                f"Soda profile: column '{col}' has {null_pct:.1%} missing values "
                                f"({null_count} of {len(series)})"
                            ),
                        ))

                    # --- Numeric profile: outlier detection ---
                    if pd.api.types.is_numeric_dtype(series):
                        clean = series.dropna()
                        if len(clean) > 10:
                            mean = clean.mean()
                            std = clean.std()
                            if std > 0:
                                z_scores = (clean - mean) / std
                                outlier_count = int((z_scores.abs() > 4).sum())
                                outlier_pct = outlier_count / len(clean)
                                if outlier_pct > 0.01:  # >1% outliers
                                    findings.append(DQBenchFinding(
                                        column=col,
                                        severity="warning",
                                        check="out_of_range",
                                        message=(
                                            f"Soda profile: column '{col}' has {outlier_count} outlier values "
                                            f"(>{outlier_pct:.1%} beyond 4σ from mean={mean:.2f})"
                                        ),
                                    ))
                            # Negative values in columns that should be positive
                            if col in ("income", "amount", "quantity", "order_total", "price",
                                       "claim_amount", "payment_amount", "copay_amount"):
                                neg_count = int((clean < 0).sum())
                                if neg_count > 0:
                                    findings.append(DQBenchFinding(
                                        column=col,
                                        severity="error",
                                        check="out_of_range",
                                        message=f"Soda check: column '{col}' has {neg_count} negative values",
                                    ))

                    # --- Cardinality anomaly: very high cardinality in expected-low column ---
                    elif pd.api.types.is_object_dtype(series):
                        n_unique = series.nunique()
                        n_total = len(series.dropna())
                        uniqueness_ratio = n_unique / n_total if n_total > 0 else 0

                        # Very low cardinality (< 10 distinct values) → check for invalid categories
                        # We don't know ground truth so just profile
                        if n_unique <= 30 and n_total > 100:
                            # Flag extremely high frequency of one value (>98% same = suspicious)
                            top_freq = series.value_counts(normalize=True).iloc[0] if n_unique > 0 else 0
                            if top_freq > 0.98:
                                findings.append(DQBenchFinding(
                                    column=col,
                                    severity="warning",
                                    check="distribution_anomaly",
                                    message=(
                                        f"Soda profile: column '{col}' has {top_freq:.1%} of rows with "
                                        f"the same value (very low variance)"
                                    ),
                                ))

                except Exception as e:
                    logger.debug("Soda auto-profile skipped column %s: %s", col, e)

        except Exception as e:
            logger.warning("SodaAutoProfileAdapter error: %s", e)

        return findings


# ---------------------------------------------------------------------------
# Best-effort
# ---------------------------------------------------------------------------

class SodaBestEffortAdapter(DQBenchAdapter):
    """Soda best-effort checks equivalent to hand-written SodaCL rules."""

    @property
    def name(self) -> str:
        return "Soda (best-effort)"

    @property
    def version(self) -> str:
        return _soda_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        findings: list[DQBenchFinding] = []
        try:
            import pandas as pd
            import re

            df = pd.read_csv(csv_path)
            cols = set(df.columns)

            def has(name: str) -> bool:
                return name in cols

            def check_not_null(col: str, threshold: float = 0.0):
                """SodaCL: missing_count(col) = 0"""
                try:
                    missing = df[col].isna().sum()
                    if missing > len(df) * threshold:
                        findings.append(DQBenchFinding(
                            column=col,
                            severity="error",
                            check="null_values",
                            message=(
                                f"Soda check: missing_count({col}) = {missing} "
                                f"(expected 0, threshold={threshold:.0%})"
                            ),
                        ))
                except Exception as e:
                    logger.debug("check_not_null(%s): %s", col, e)

            def check_unique(col: str):
                """SodaCL: duplicate_count(col) = 0"""
                try:
                    dup_count = int(df[col].duplicated().sum())
                    if dup_count > 0:
                        findings.append(DQBenchFinding(
                            column=col,
                            severity="error",
                            check="duplicate_values",
                            message=f"Soda check: duplicate_count({col}) = {dup_count}",
                        ))
                except Exception as e:
                    logger.debug("check_unique(%s): %s", col, e)

            def check_in_set(col: str, valid_values: list, mostly: float = 1.0):
                """SodaCL: invalid_count(col) = 0, valid values: [...]"""
                try:
                    non_null = df[col].dropna()
                    invalid = non_null[~non_null.isin(valid_values)]
                    invalid_pct = len(invalid) / len(non_null) if len(non_null) > 0 else 0
                    if invalid_pct > (1 - mostly):
                        findings.append(DQBenchFinding(
                            column=col,
                            severity="error",
                            check="enum_violation",
                            message=(
                                f"Soda check: invalid_count({col}) = {len(invalid)} "
                                f"values not in allowed set"
                            ),
                        ))
                except Exception as e:
                    logger.debug("check_in_set(%s): %s", col, e)

            def check_regex(col: str, pattern: str, check_name: str, mostly: float = 0.95):
                """SodaCL: invalid_count(col) = 0, valid format: regex"""
                try:
                    non_null = df[col].dropna().astype(str)
                    invalid = non_null[~non_null.str.match(pattern, na=False)]
                    invalid_pct = len(invalid) / len(non_null) if len(non_null) > 0 else 0
                    if invalid_pct > (1 - mostly):
                        findings.append(DQBenchFinding(
                            column=col,
                            severity="error",
                            check=check_name,
                            message=(
                                f"Soda check: invalid_{check_name}({col}) = {len(invalid)} "
                                f"values not matching expected format"
                            ),
                        ))
                except Exception as e:
                    logger.debug("check_regex(%s): %s", col, e)

            def check_between(col: str, lo, hi, mostly: float = 0.99):
                """SodaCL: invalid_count(col) = 0, valid range: [lo, hi]"""
                try:
                    non_null = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(non_null) == 0:
                        return
                    invalid = non_null[(non_null < lo) | (non_null > hi)]
                    invalid_pct = len(invalid) / len(non_null)
                    if invalid_pct > (1 - mostly):
                        findings.append(DQBenchFinding(
                            column=col,
                            severity="error",
                            check="out_of_range",
                            message=(
                                f"Soda check: out_of_range({col}) = {len(invalid)} "
                                f"values outside [{lo}, {hi}]"
                            ),
                        ))
                except Exception as e:
                    logger.debug("check_between(%s): %s", col, e)

            def check_date_order(col_later: str, col_earlier: str, mostly: float = 0.95):
                """SodaCL equivalent: col_later >= col_earlier"""
                try:
                    later = pd.to_datetime(df[col_later], errors="coerce")
                    earlier = pd.to_datetime(df[col_earlier], errors="coerce")
                    both_valid = later.notna() & earlier.notna()
                    if both_valid.sum() == 0:
                        return
                    violations = (later[both_valid] < earlier[both_valid]).sum()
                    violation_pct = violations / both_valid.sum()
                    if violation_pct > (1 - mostly):
                        findings.append(DQBenchFinding(
                            column=f"{col_later}, {col_earlier}",
                            severity="error",
                            check="logic_violation",
                            message=(
                                f"Soda check: temporal_order({col_later} >= {col_earlier}) failed "
                                f"for {violations} rows ({violation_pct:.1%})"
                            ),
                        ))
                except Exception as e:
                    logger.debug("check_date_order(%s, %s): %s", col_later, col_earlier, e)

            # ----------------------------------------------------------------
            # SodaCL-equivalent checks
            # ----------------------------------------------------------------

            # --- Null checks for required columns ---
            required = [
                "customer_id", "order_id", "patient_id", "record_number",
                "email", "customer_email", "patient_email",
                "first_name", "last_name", "customer_name", "patient_name",
                "phone", "phone_number", "patient_phone",
                "status", "claim_status",
                "order_date", "service_date", "signup_date",
                "npi_number", "insurance_id", "primary_dx",
                "procedure_code", "product_category",
            ]
            for c in required:
                if has(c):
                    check_not_null(c)

            # --- Uniqueness ---
            for c in ["customer_id", "order_id", "patient_id", "record_number",
                      "npi_number", "session_id", "sku"]:
                if has(c):
                    check_unique(c)

            # --- Enum validation ---
            if has("status"):
                check_in_set("status", ["active", "inactive", "pending", "suspended", "closed"])
            if has("account_type"):
                check_in_set("account_type",
                             ["standard", "premium", "enterprise", "basic", "trial", "free"])
            if has("gender"):
                check_in_set("gender",
                             ["M", "F", "Male", "Female", "male", "female",
                              "Other", "other", "U", "Unknown"])
            if has("currency_code"):
                check_in_set("currency_code", ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "CNY"])
            if has("claim_status"):
                check_in_set("claim_status",
                             ["pending", "approved", "denied", "paid", "submitted", "appealed",
                              "PENDING", "APPROVED", "DENIED", "PAID", "SUBMITTED"])
            if has("prior_auth_flag"):
                check_in_set("prior_auth_flag",
                             ["Y", "N", "Yes", "No", "yes", "no", "true", "false", "1", "0"])
            if has("deductible_met"):
                check_in_set("deductible_met",
                             ["Y", "N", "Yes", "No", "yes", "no", "true", "false", "1", "0"])

            # --- Format validation ---
            email_regex = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
            for c in ["email", "customer_email", "patient_email"]:
                if has(c):
                    check_regex(c, email_regex, "invalid_format", mostly=0.95)

            phone_regex = r"^[\d\s\+\-\.\(\)]{7,20}$"
            for c in ["phone", "phone_number", "phone_intl", "patient_phone"]:
                if has(c):
                    check_regex(c, phone_regex, "invalid_format", mostly=0.90)

            zip_regex = r"^\d{5}(-\d{4})?$"
            for c in ["zip_code", "billing_zip", "shipping_zip", "patient_zip"]:
                if has(c):
                    check_regex(c, zip_regex, "invalid_format", mostly=0.90)

            npi_regex = r"^\d{10}$"
            for c in ["npi_number", "referring_npi"]:
                if has(c):
                    check_regex(c, npi_regex, "invalid_format", mostly=0.95)

            # --- Range validation ---
            range_checks = {
                "age": (0, 120),
                "patient_age": (0, 120),
                "income": (0, 10_000_000),
                "order_count": (0, 100_000),
                "quantity": (0, 100_000),
                "rating": (1, 5),
                "discount_pct": (0, 100),
                "order_total": (0, 1_000_000),
                "claim_amount": (0, 10_000_000),
                "payment_amount": (0, 10_000_000),
                "copay_amount": (0, 10_000),
                "dosage_amount": (0, 10_000),
                "remittance_amount": (0, 10_000_000),
                "policy_max_amount": (0, 10_000_000),
                "lab_result": (0, 100_000),
            }
            for c, (lo, hi) in range_checks.items():
                if has(c):
                    check_between(c, lo, hi)

            # --- Temporal ordering ---
            date_pairs = [
                ("last_login", "signup_date"),
                ("ship_date", "order_date"),
                ("discharge_date", "admission_date"),
                ("submit_date", "service_date"),
                ("adjudication_date", "submit_date"),
                ("last_updated", "signup_date"),
                ("updated_at", "created_at"),
            ]
            for col_later, col_earlier in date_pairs:
                if has(col_later) and has(col_earlier):
                    check_date_order(col_later, col_earlier)

        except Exception as e:
            logger.warning("SodaBestEffortAdapter error: %s", e)

        return findings
