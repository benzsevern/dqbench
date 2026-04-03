"""Pandera adapters for DQBench — zero-config, auto-profiled, best-effort."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding

logger = logging.getLogger(__name__)


def _pandera_version() -> str:
    try:
        import pandera
        return pandera.__version__
    except Exception:
        return "unknown"


def _failure_cases_to_findings(failure_cases, check_name: str = "schema_check") -> list[DQBenchFinding]:
    """Convert a pandera failure_cases DataFrame into DQBenchFindings."""
    findings: list[DQBenchFinding] = []
    try:
        if failure_cases is None or len(failure_cases) == 0:
            return findings

        # Group by column and schema_context to deduplicate
        grouped = {}
        for _, row in failure_cases.iterrows():
            try:
                col = str(row.get("column", row.get("schema_context", "unknown")))
                check = str(row.get("check", check_name))
                failure_case = row.get("failure_case", None)
                key = (col, check)
                if key not in grouped:
                    grouped[key] = []
                if failure_case is not None:
                    grouped[key].append(str(failure_case)[:50])
            except Exception:
                pass

        for (col, check), cases in grouped.items():
            sample = ", ".join(cases[:3])
            msg_parts = [f"Pandera check failed: '{check}' on column '{col}'"]
            if sample:
                msg_parts.append(f"(sample failures: {sample})")
            message = " ".join(msg_parts)

            # Map pandera check names to DQBench categories
            check_category = _map_pandera_check(check)
            findings.append(DQBenchFinding(
                column=col,
                severity="error",
                check=check_category,
                message=message,
            ))
    except Exception as e:
        logger.warning("Could not parse pandera failure_cases: %s", e)
    return findings


def _map_pandera_check(check: str) -> str:
    """Map pandera check name to DQBench check category."""
    check_lower = check.lower()
    if "not_nullable" in check_lower or "null" in check_lower:
        return "null_values"
    if "unique" in check_lower or "duplicate" in check_lower:
        return "duplicate_values"
    if "isin" in check_lower or "in_set" in check_lower or "in(" in check_lower:
        return "enum_violation"
    if "str_matches" in check_lower or "regex" in check_lower or "str_contains" in check_lower:
        return "invalid_format"
    if "between" in check_lower or "less_than" in check_lower or "greater_than" in check_lower:
        return "out_of_range"
    if "dtype" in check_lower or "type" in check_lower:
        return "wrong_dtype"
    if "str_length" in check_lower or "length" in check_lower:
        return "invalid_format"
    return check


# ---------------------------------------------------------------------------
# Zero-config
# ---------------------------------------------------------------------------

class PanderaZeroConfigAdapter(DQBenchAdapter):
    """Pandera with no schema — returns empty findings."""

    @property
    def name(self) -> str:
        return "Pandera (zero-config)"

    @property
    def version(self) -> str:
        return _pandera_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        # No schema = no checks = no findings
        return []


# ---------------------------------------------------------------------------
# Auto-profiled
# ---------------------------------------------------------------------------

class PanderaAutoProfileAdapter(DQBenchAdapter):
    """Pandera with infer_schema() — validates against observed data patterns."""

    @property
    def name(self) -> str:
        return "Pandera (auto-profiled)"

    @property
    def version(self) -> str:
        return _pandera_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        findings: list[DQBenchFinding] = []
        try:
            import pandas as pd
            import pandera as pa

            df = pd.read_csv(csv_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                schema = pa.infer_schema(df)

            try:
                schema.validate(df, lazy=True)
                # No errors — schema matches perfectly (expected for auto-profiled on same data)
            except pa.errors.SchemaErrors as err:
                findings = _failure_cases_to_findings(err.failure_cases, "inferred_schema")
            except pa.errors.SchemaError as err:
                # Single error
                try:
                    col = str(err.schema.name) if hasattr(err.schema, "name") else "unknown"
                    findings.append(DQBenchFinding(
                        column=col,
                        severity="error",
                        check=_map_pandera_check(str(type(err).__name__)),
                        message=str(err)[:200],
                    ))
                except Exception:
                    pass
            except Exception as e:
                logger.debug("Pandera validation error: %s", e)

        except Exception as e:
            logger.warning("PanderaAutoProfileAdapter error: %s", e)

        return findings


# ---------------------------------------------------------------------------
# Best-effort
# ---------------------------------------------------------------------------

class PanderaBestEffortAdapter(DQBenchAdapter):
    """Pandera with hand-crafted DataFrameSchema based on column names."""

    @property
    def name(self) -> str:
        return "Pandera (best-effort)"

    @property
    def version(self) -> str:
        return _pandera_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        findings: list[DQBenchFinding] = []
        try:
            import pandas as pd
            import pandera as pa
            from pandera import Column, Check, DataFrameSchema

            df = pd.read_csv(csv_path)
            columns_present = set(df.columns)
            schema_cols = {}

            def col(name: str, **kwargs) -> bool:
                return name in columns_present

            # ----------------------------------------------------------------
            # Build column specs
            # ----------------------------------------------------------------

            # ID columns — not null, unique, positive int
            for c in ["customer_id", "order_id", "patient_id", "record_number"]:
                if col(c):
                    schema_cols[c] = Column(
                        nullable=False,
                        checks=[Check.greater_than(0)],
                        coerce=True,
                    )

            # Email — not null, matches email pattern
            email_regex = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
            for c in ["email", "customer_email", "patient_email"]:
                if col(c):
                    schema_cols[c] = Column(
                        str,
                        checks=[
                            Check.str_matches(email_regex, error="invalid email format"),
                        ],
                        nullable=True,
                    )

            # Phone — matches phone pattern
            phone_regex = r"^[\d\s\+\-\.\(\)]{7,20}$"
            for c in ["phone", "phone_number", "phone_intl", "patient_phone"]:
                if col(c):
                    schema_cols[c] = Column(
                        str,
                        checks=[
                            Check.str_matches(phone_regex, error="invalid phone format"),
                        ],
                        nullable=True,
                    )

            # Status enums
            if col("status"):
                schema_cols["status"] = Column(
                    str,
                    checks=[
                        Check.isin(["active", "inactive", "pending", "suspended", "closed"]),
                    ],
                    nullable=False,
                )

            if col("account_type"):
                schema_cols["account_type"] = Column(
                    str,
                    checks=[
                        Check.isin(["standard", "premium", "enterprise", "basic", "trial", "free"]),
                    ],
                    nullable=True,
                )

            if col("gender"):
                schema_cols["gender"] = Column(
                    str,
                    checks=[
                        Check.isin(["M", "F", "Male", "Female", "male", "female",
                                    "Other", "other", "U", "Unknown"]),
                    ],
                    nullable=True,
                )

            if col("currency_code"):
                schema_cols["currency_code"] = Column(
                    str,
                    checks=[Check.isin(["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "CNY"])],
                    nullable=True,
                )

            if col("claim_status"):
                schema_cols["claim_status"] = Column(
                    str,
                    checks=[
                        Check.isin(["pending", "approved", "denied", "paid", "submitted", "appealed",
                                    "PENDING", "APPROVED", "DENIED", "PAID", "SUBMITTED"]),
                    ],
                    nullable=True,
                )

            if col("prior_auth_flag"):
                schema_cols["prior_auth_flag"] = Column(
                    str,
                    checks=[Check.isin(["Y", "N", "Yes", "No", "yes", "no", "true", "false", "1", "0"])],
                    nullable=True,
                )

            if col("deductible_met"):
                schema_cols["deductible_met"] = Column(
                    str,
                    checks=[Check.isin(["Y", "N", "Yes", "No", "yes", "no", "true", "false", "1", "0"])],
                    nullable=True,
                )

            # Numeric ranges
            numeric_ranges = {
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
            for c, (lo, hi) in numeric_ranges.items():
                if col(c):
                    schema_cols[c] = Column(
                        nullable=True,
                        checks=[
                            Check.greater_than_or_equal_to(lo),
                            Check.less_than_or_equal_to(hi),
                        ],
                        coerce=True,
                    )

            # ZIP code format
            zip_regex = r"^\d{5}(-\d{4})?$"
            for c in ["zip_code", "billing_zip", "shipping_zip", "patient_zip"]:
                if col(c):
                    schema_cols[c] = Column(
                        str,
                        checks=[Check.str_matches(zip_regex, error="invalid zip format")],
                        nullable=True,
                        coerce=True,
                    )

            # NPI format (10 digits)
            npi_regex = r"^\d{10}$"
            for c in ["npi_number", "referring_npi"]:
                if col(c):
                    schema_cols[c] = Column(
                        str,
                        checks=[Check.str_matches(npi_regex, error="invalid NPI format")],
                        nullable=True,
                        coerce=True,
                    )

            # Required non-null columns
            required_not_null = [
                "first_name", "last_name", "customer_name", "patient_name",
                "order_date", "service_date", "signup_date",
                "primary_dx", "insurance_id", "procedure_code",
                "product_category", "country",
            ]
            for c in required_not_null:
                if col(c) and c not in schema_cols:
                    schema_cols[c] = Column(nullable=False)

            # ----------------------------------------------------------------
            # Validate
            # ----------------------------------------------------------------
            if not schema_cols:
                return findings

            schema = DataFrameSchema(
                columns=schema_cols,
                strict=False,  # Don't fail on extra columns
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    schema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as err:
                findings = _failure_cases_to_findings(err.failure_cases, "best_effort_check")
            except pa.errors.SchemaError as err:
                try:
                    col_name = str(getattr(err, "schema", {}).get("name", "unknown"))
                    findings.append(DQBenchFinding(
                        column=col_name,
                        severity="error",
                        check="schema_check",
                        message=str(err)[:200],
                    ))
                except Exception:
                    pass
            except Exception as e:
                logger.debug("Pandera best-effort validation error: %s", e)

        except Exception as e:
            logger.warning("PanderaBestEffortAdapter error: %s", e)

        return findings
