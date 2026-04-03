"""Great Expectations adapters for DQBench — zero-config, auto-profiled, best-effort."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gx_version() -> str:
    try:
        import great_expectations as gx
        return gx.__version__
    except Exception:
        return "unknown"


def _result_to_findings(result, csv_path: Path) -> list[DQBenchFinding]:
    """Convert a GX ExpectationSuiteValidationResult into DQBenchFindings."""
    findings: list[DQBenchFinding] = []
    try:
        for r in result.results:
            if r.success:
                continue
            try:
                cfg = r.expectation_config
                exp_type = cfg.type if hasattr(cfg, "type") else str(cfg.get("expectation_type", "unknown"))
                col = "unknown"
                try:
                    kw = cfg.kwargs if hasattr(cfg, "kwargs") else {}
                    col = kw.get("column") or kw.get("column_A") or kw.get("column_list") or "dataset"
                    if isinstance(col, list):
                        col = ", ".join(col)
                    col = str(col)
                except Exception:
                    pass

                # Map expectation type to meaningful check/severity
                severity = "error"
                message = f"GX expectation failed: {exp_type}"

                try:
                    res_dict = r.result or {}
                    unexpected_pct = res_dict.get("unexpected_percent", None)
                    if unexpected_pct is not None:
                        message = f"{exp_type}: {unexpected_pct:.1f}% unexpected values in column '{col}'"
                    else:
                        message = f"{exp_type} failed for column '{col}'"
                except Exception:
                    pass

                findings.append(DQBenchFinding(
                    column=col,
                    severity=severity,
                    check=_exp_type_to_check(exp_type),
                    message=message,
                ))
            except Exception as e:
                logger.debug("Could not parse GX result row: %s", e)
    except Exception as e:
        logger.warning("Could not iterate GX results: %s", e)
    return findings


def _exp_type_to_check(exp_type: str) -> str:
    """Map GX expectation type strings to DQBench check categories."""
    mapping = {
        "expect_column_values_to_not_be_null": "null_values",
        "expect_column_values_to_be_unique": "duplicate_values",
        "expect_column_values_to_be_in_set": "enum_violation",
        "expect_column_values_to_match_regex": "invalid_format",
        "expect_column_values_to_be_between": "out_of_range",
        "expect_column_pair_values_a_to_be_greater_than_b": "logic_violation",
        "expect_column_values_to_be_of_type": "wrong_dtype",
        "expect_column_values_to_be_in_type_list": "wrong_dtype",
        "expect_column_values_to_match_strftime_format": "invalid_format",
        "expect_column_values_to_not_match_regex": "invalid_format",
        "expect_table_row_count_to_be_between": "out_of_range",
    }
    key = exp_type.lower()
    return mapping.get(key, exp_type.replace("expect_", "").replace("_", " "))


def _build_gx_context_and_batch(csv_path: Path):
    """Build an ephemeral GX context and batch definition for the given CSV."""
    import great_expectations as gx
    ctx = gx.get_context(mode="ephemeral")
    ds = ctx.data_sources.add_pandas(name="bench")
    asset = ds.add_csv_asset(name="data", filepath_or_buffer=str(csv_path))
    batch_def = asset.add_batch_definition_whole_dataframe("whole_df")
    return ctx, batch_def


# ---------------------------------------------------------------------------
# Zero-config
# ---------------------------------------------------------------------------

class GXZeroConfigAdapter(DQBenchAdapter):
    """GX with no expectations configured — returns empty findings."""

    @property
    def name(self) -> str:
        return "GX (zero-config)"

    @property
    def version(self) -> str:
        return _gx_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        # GX cannot validate without expectations — return empty
        return []


# ---------------------------------------------------------------------------
# Auto-profiled
# ---------------------------------------------------------------------------

class GXAutoProfileAdapter(DQBenchAdapter):
    """GX with auto-generated expectations using OnboardingDataAssistant."""

    @property
    def name(self) -> str:
        return "GX (auto-profiled)"

    @property
    def version(self) -> str:
        return _gx_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        findings: list[DQBenchFinding] = []
        try:
            import pandas as pd
            import great_expectations as gx
            from great_expectations import expectations as gxe

            df = pd.read_csv(csv_path)

            # Build context
            ctx, batch_def = _build_gx_context_and_batch(csv_path)
            suite = ctx.suites.add(gx.ExpectationSuite(name="auto_suite"))

            # Auto-profile: add expectations based on observed data statistics
            # This is the closest GX has to auto-profiling without a cloud backend
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for col in df.columns:
                    try:
                        # Always check not-null if column has low null rate
                        null_rate = df[col].isna().mean()
                        if null_rate < 0.01:  # Less than 1% nulls → expect no nulls
                            suite.add_expectation(
                                gxe.ExpectColumnValuesToNotBeNull(column=col)
                            )
                        # Type-based expectations
                        dtype = df[col].dtype
                        if dtype in ("int64", "float64"):
                            col_min = float(df[col].dropna().min())
                            col_max = float(df[col].dropna().max())
                            # Allow 20% slack beyond observed range
                            slack = max(abs(col_max - col_min) * 0.20, 1)
                            suite.add_expectation(
                                gxe.ExpectColumnValuesToBeBetween(
                                    column=col,
                                    min_value=col_min - slack,
                                    max_value=col_max + slack,
                                )
                            )
                        elif dtype == "object":
                            n_unique = df[col].nunique()
                            n_rows = len(df)
                            # Categorical: low cardinality columns
                            if n_unique <= 20 and n_unique / n_rows < 0.05:
                                value_set = sorted(
                                    [str(v) for v in df[col].dropna().unique()]
                                )
                                suite.add_expectation(
                                    gxe.ExpectColumnValuesToBeInSet(
                                        column=col, value_set=value_set
                                    )
                                )
                    except Exception as e:
                        logger.debug("Auto-profile skipped column %s: %s", col, e)

            # Run validation
            vd = ctx.validation_definitions.add(
                gx.ValidationDefinition(name="auto_vd", data=batch_def, suite=suite)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = vd.run()

            findings = _result_to_findings(result, csv_path)

        except Exception as e:
            logger.warning("GXAutoProfileAdapter error: %s", e)

        return findings


# ---------------------------------------------------------------------------
# Best-effort
# ---------------------------------------------------------------------------

class GXBestEffortAdapter(DQBenchAdapter):
    """GX with hand-crafted expectations based on column names."""

    @property
    def name(self) -> str:
        return "GX (best-effort)"

    @property
    def version(self) -> str:
        return _gx_version()

    def validate(self, csv_path: Path) -> list[DQBenchFinding]:
        findings: list[DQBenchFinding] = []
        try:
            import pandas as pd
            import great_expectations as gx
            from great_expectations import expectations as gxe

            df = pd.read_csv(csv_path)
            columns = set(df.columns)

            ctx, batch_def = _build_gx_context_and_batch(csv_path)
            suite = ctx.suites.add(gx.ExpectationSuite(name="best_effort_suite"))

            def add(exp):
                try:
                    suite.add_expectation(exp)
                except Exception as e:
                    logger.debug("Could not add expectation: %s", e)

            def has(name: str) -> bool:
                return name in columns

            # ----------------------------------------------------------------
            # Null checks on commonly required columns
            # ----------------------------------------------------------------
            required_cols = [
                "customer_id", "order_id", "patient_id", "record_number",
                "email", "customer_email", "patient_email",
                "first_name", "last_name", "customer_name", "patient_name",
                "phone", "phone_number", "patient_phone",
                "status", "claim_status",
                "order_date", "service_date", "signup_date",
                "npi_number", "insurance_id", "primary_dx",
            ]
            for col in required_cols:
                if has(col):
                    add(gxe.ExpectColumnValuesToNotBeNull(column=col))

            # ----------------------------------------------------------------
            # Uniqueness checks for ID columns
            # ----------------------------------------------------------------
            id_cols = [
                "customer_id", "order_id", "patient_id", "record_number",
                "npi_number", "session_id", "sku",
            ]
            for col in id_cols:
                if has(col):
                    add(gxe.ExpectColumnValuesToBeUnique(column=col))

            # ----------------------------------------------------------------
            # Enum / categorical checks
            # ----------------------------------------------------------------
            if has("status"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="status",
                    value_set=["active", "inactive", "pending", "suspended", "closed"],
                ))
            if has("account_type"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="account_type",
                    value_set=["standard", "premium", "enterprise", "basic", "trial", "free"],
                ))
            if has("gender"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="gender",
                    value_set=["M", "F", "Male", "Female", "male", "female", "Other", "other", "U", "Unknown"],
                ))
            if has("product_category"):
                # Just check not null and reasonable cardinality
                add(gxe.ExpectColumnValuesToNotBeNull(column="product_category"))
            if has("currency_code"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="currency_code",
                    value_set=["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "CNY"],
                ))
            if has("country"):
                add(gxe.ExpectColumnValuesToNotBeNull(column="country"))
            if has("claim_status"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="claim_status",
                    value_set=["pending", "approved", "denied", "paid", "submitted", "appealed",
                               "PENDING", "APPROVED", "DENIED", "PAID", "SUBMITTED"],
                ))
            if has("service_type"):
                add(gxe.ExpectColumnValuesToNotBeNull(column="service_type"))
            if has("provider_type"):
                add(gxe.ExpectColumnValuesToNotBeNull(column="provider_type"))
            if has("prior_auth_flag"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="prior_auth_flag",
                    value_set=["Y", "N", "Yes", "No", "yes", "no", "true", "false", "1", "0"],
                ))
            if has("deductible_met"):
                add(gxe.ExpectColumnValuesToBeInSet(
                    column="deductible_met",
                    value_set=["Y", "N", "Yes", "No", "yes", "no", "true", "false", "1", "0"],
                ))

            # ----------------------------------------------------------------
            # Email format
            # ----------------------------------------------------------------
            email_regex = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
            for col in ["email", "customer_email", "patient_email"]:
                if has(col):
                    add(gxe.ExpectColumnValuesToMatchRegex(
                        column=col,
                        regex=email_regex,
                        mostly=0.95,
                    ))

            # ----------------------------------------------------------------
            # Phone format
            # ----------------------------------------------------------------
            phone_regex = r"^[\d\s\+\-\.\(\)]{7,20}$"
            for col in ["phone", "phone_number", "phone_intl", "patient_phone", "emergency_contact"]:
                if has(col):
                    add(gxe.ExpectColumnValuesToMatchRegex(
                        column=col,
                        regex=phone_regex,
                        mostly=0.90,
                    ))

            # ----------------------------------------------------------------
            # Numeric range checks
            # ----------------------------------------------------------------
            if has("age"):
                add(gxe.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=120, mostly=0.99))
            if has("patient_age"):
                add(gxe.ExpectColumnValuesToBeBetween(column="patient_age", min_value=0, max_value=120, mostly=0.99))
            if has("income"):
                add(gxe.ExpectColumnValuesToBeBetween(column="income", min_value=0, max_value=10_000_000, mostly=0.99))
            if has("order_count"):
                add(gxe.ExpectColumnValuesToBeBetween(column="order_count", min_value=0, max_value=100_000, mostly=0.99))
            if has("quantity"):
                add(gxe.ExpectColumnValuesToBeBetween(column="quantity", min_value=0, max_value=100_000, mostly=0.99))
            if has("rating"):
                add(gxe.ExpectColumnValuesToBeBetween(column="rating", min_value=1, max_value=5, mostly=0.99))
            if has("discount_pct"):
                add(gxe.ExpectColumnValuesToBeBetween(column="discount_pct", min_value=0, max_value=100, mostly=0.99))
            if has("order_total"):
                add(gxe.ExpectColumnValuesToBeBetween(column="order_total", min_value=0, max_value=1_000_000, mostly=0.99))
            if has("claim_amount"):
                add(gxe.ExpectColumnValuesToBeBetween(column="claim_amount", min_value=0, max_value=10_000_000, mostly=0.99))
            if has("payment_amount"):
                add(gxe.ExpectColumnValuesToBeBetween(column="payment_amount", min_value=0, max_value=10_000_000, mostly=0.99))
            if has("copay_amount"):
                add(gxe.ExpectColumnValuesToBeBetween(column="copay_amount", min_value=0, max_value=10_000, mostly=0.99))
            if has("dosage_amount"):
                add(gxe.ExpectColumnValuesToBeBetween(column="dosage_amount", min_value=0, max_value=10_000, mostly=0.99))
            if has("remittance_amount"):
                add(gxe.ExpectColumnValuesToBeBetween(column="remittance_amount", min_value=0, max_value=10_000_000, mostly=0.99))
            if has("policy_max_amount"):
                add(gxe.ExpectColumnValuesToBeBetween(column="policy_max_amount", min_value=0, max_value=10_000_000, mostly=0.99))

            # ----------------------------------------------------------------
            # Date ordering checks (column_pair_A > column_pair_B)
            # ----------------------------------------------------------------
            date_pair_checks = [
                ("last_login", "signup_date"),      # last_login >= signup_date
                ("ship_date", "order_date"),         # ship_date >= order_date
                ("discharge_date", "admission_date"), # discharge >= admission
                ("submit_date", "service_date"),     # submit >= service
                ("adjudication_date", "submit_date"), # adjudication >= submit
                ("last_updated", "signup_date"),
                ("updated_at", "created_at"),
            ]
            for col_a, col_b in date_pair_checks:
                if has(col_a) and has(col_b):
                    # We compare as strings (ISO dates sort lexicographically)
                    add(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                        column_A=col_a,
                        column_B=col_b,
                        or_equal=True,
                        mostly=0.95,
                    ))

            # ----------------------------------------------------------------
            # ZIP / postal code format
            # ----------------------------------------------------------------
            zip_regex = r"^\d{5}(-\d{4})?$"
            for col in ["zip_code", "billing_zip", "shipping_zip", "patient_zip"]:
                if has(col):
                    add(gxe.ExpectColumnValuesToMatchRegex(
                        column=col,
                        regex=zip_regex,
                        mostly=0.90,
                    ))

            # ----------------------------------------------------------------
            # NPI number format (10 digits)
            # ----------------------------------------------------------------
            npi_regex = r"^\d{10}$"
            for col in ["npi_number", "referring_npi"]:
                if has(col):
                    add(gxe.ExpectColumnValuesToMatchRegex(
                        column=col,
                        regex=npi_regex,
                        mostly=0.95,
                    ))

            # ----------------------------------------------------------------
            # Run validation
            # ----------------------------------------------------------------
            vd = ctx.validation_definitions.add(
                gx.ValidationDefinition(name="best_effort_vd", data=batch_def, suite=suite)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = vd.run()

            findings = _result_to_findings(result, csv_path)

        except Exception as e:
            logger.warning("GXBestEffortAdapter error: %s", e)

        return findings
