"""Compute benchmark scores from findings vs ground truth."""
from __future__ import annotations
from dqbench.models import DQBenchFinding, TierResult
from dqbench.ground_truth import GroundTruth

# ---------------------------------------------------------------------------
# Keyword sets for fuzzy issue-type matching
# ---------------------------------------------------------------------------
ISSUE_KEYWORDS: dict[str, list[str]] = {
    "uniqueness": ["unique", "duplicate", "primary key", "distinct"],
    "duplicate_values": ["unique", "duplicate", "primary key", "distinct"],
    "nullability": ["null", "missing", "required", "empty"],
    "null_values": ["null", "missing", "required", "empty"],
    "format_detection": ["format", "email", "phone", "url", "pattern", "invalid"],
    "invalid_format": ["format", "email", "phone", "url", "pattern", "invalid"],
    "inconsistent_format": ["format", "pattern", "inconsistent", "mixed", "invalid"],
    "type_inference": ["type", "numeric", "integer", "string", "cast"],
    "wrong_type": ["type", "numeric", "integer", "string", "cast", "invalid"],
    "wrong_dtype": ["type", "dtype", "integer", "numeric", "cast", "schema"],
    "range_distribution": ["range", "outlier", "stddev", "min", "max", "extreme"],
    "outlier_values": ["outlier", "range", "extreme", "stddev", "min", "max"],
    "out_of_range": ["range", "outlier", "extreme", "min", "max", "exceed", "invalid"],
    "cardinality": ["enum", "cardinality", "unique values", "categorical"],
    "enum_violation": ["enum", "cardinality", "categorical", "allowed", "invalid value"],
    "pattern_consistency": ["pattern", "inconsistent", "format", "mixed"],
    "temporal_order": ["temporal", "order", "before", "after", "date"],
    "logic_violation": ["before", "after", "mismatch", "inconsistent", "invalid", "exceed", "logic", "violat"],
    "null_correlation": ["correlat", "null group", "null together"],
    "invalid_values": ["invalid", "range", "negative", "illegal", "bad value"],
    "misspelled_values": ["misspell", "typo", "invalid", "pattern", "inconsistent"],
    "distribution_shift": ["drift", "new categor", "distribution change", "shift", "new value"],
    "distribution_anomaly": ["distribution", "anomaly", "bimodal", "gap", "outlier"],
    "mixed_coding_standard": ["standard", "format", "icd", "coding", "version", "invalid"],
    "sequence_gap": ["sequence", "gap", "missing", "sequential", "order"],
    # Adversarial-specific (Tier 3)
    "checksum_failure": ["luhn", "check digit", "checksum", "npi", "invalid number"],
    "luhn": ["luhn", "check digit", "checksum", "npi", "invalid number"],
    "cross_column": ["mismatch", "doesn't match", "inconsistent with", "wrong for", "exceeds"],
    "encoding": ["encoding", "unicode", "utf", "latin", "character", "zero-width", "smart quote", "latin-1", "latin1"],
    "encoding_issue": ["encoding", "unicode", "utf", "latin", "character", "zero-width", "smart quote", "latin-1", "latin1"],
    "invisible_chars": ["zero-width", "invisible", "unicode", "encoding", "character"],
    "inconsistent_encoding": ["encoding", "unicode", "quote", "smart quote", "curly", "character"],
    "semantic": ["semantic", "domain", "invalid value", "impossible", "weekend", "negative"],
    "drift": ["drift", "new categor", "distribution change", "shift"],
}


def _finding_matches_issue(finding: DQBenchFinding, issue_type: str) -> bool:
    """Return True if the finding matches the planted issue type.

    Matching rules (either is sufficient):
    1. finding.check exactly equals the issue_type, OR
    2. finding.check or finding.message (lowercased) contains any keyword
       from the issue_type's keyword set.
    """
    # Rule 1: exact check match
    if finding.check.lower() == issue_type.lower():
        return True

    # Rule 2: keyword match
    keywords = ISSUE_KEYWORDS.get(issue_type, [])
    check_lower = finding.check.lower()
    message_lower = finding.message.lower()
    for kw in keywords:
        if kw in check_lower or kw in message_lower:
            return True

    return False


def score_tier(
    findings: list[DQBenchFinding],
    ground_truth: GroundTruth,
    tier: int,
    time_seconds: float,
    memory_mb: float,
) -> TierResult:
    planted_cols = set(ground_truth.planted_columns.keys())
    clean_cols = set(ground_truth.clean_columns)

    # ------------------------------------------------------------------
    # Column-level scoring (existing logic)
    # ------------------------------------------------------------------

    # Detection: any finding on a planted column (any severity)
    detected_cols = set()
    for f in findings:
        for part in f.column.split(","):
            col = part.strip()
            if col in planted_cols:
                detected_cols.add(col)

    # False positives: WARNING/ERROR on clean columns (INFO is NOT a FP)
    fp_cols = set()
    for f in findings:
        if f.severity.lower() in ("error", "warning"):
            for part in f.column.split(","):
                col = part.strip()
                if col in clean_cols:
                    fp_cols.add(col)

    tp = len(detected_cols)
    fp = len(fp_cols)

    recall = tp / len(planted_cols) if planted_cols else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = len(fp_cols) / len(clean_cols) if clean_cols else 0.0

    # ------------------------------------------------------------------
    # Issue-level scoring (NEW)
    # ------------------------------------------------------------------

    # Build the set of planted (column, issue_type) pairs
    planted_issues: set[tuple[str, str]] = set()
    for col, planted_col in ground_truth.planted_columns.items():
        for issue in planted_col.issues:
            planted_issues.add((col, issue))

    # Determine which planted issues were detected
    detected_issues: set[tuple[str, str]] = set()
    # Count findings that match at least one planted issue (for precision)
    matched_finding_count = 0

    for f in findings:
        f_cols = [part.strip() for part in f.column.split(",")]
        finding_matched = False
        for col in f_cols:
            if col in planted_cols:
                # Check if this finding matches any specific planted issue for this col
                planted_col_issues = ground_truth.planted_columns[col].issues
                for issue_type in planted_col_issues:
                    if _finding_matches_issue(f, issue_type):
                        detected_issues.add((col, issue_type))
                        finding_matched = True
        if finding_matched:
            matched_finding_count += 1

    n_planted_issues = len(planted_issues)
    n_detected_issues = len(detected_issues)

    # Total issue-flagging findings (for precision denominator):
    # any finding on a planted column that has ERROR/WARNING severity counts
    # as an "issue assertion" (we're generous here — only penalize for
    # findings that land on clean columns or planted-but-wrong-type columns)
    issue_tp = n_detected_issues
    # Issue precision: among all findings that target planted columns
    # (as warnings/errors), what fraction match a planted issue?
    planted_col_findings = [
        f for f in findings
        if f.severity.lower() in ("error", "warning")
        and any(part.strip() in planted_cols for part in f.column.split(","))
    ]
    issue_precision_denom = len(planted_col_findings) + fp  # planted + FP on clean

    issue_recall = n_detected_issues / n_planted_issues if n_planted_issues else 0.0
    issue_precision = (
        issue_tp / issue_precision_denom if issue_precision_denom > 0 else 0.0
    )
    issue_f1 = (
        2 * issue_precision * issue_recall / (issue_precision + issue_recall)
        if (issue_precision + issue_recall) > 0
        else 0.0
    )

    return TierResult(
        tier=tier,
        recall=round(recall, 4),
        precision=round(precision, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        issue_recall=round(issue_recall, 4),
        issue_precision=round(issue_precision, 4),
        issue_f1=round(issue_f1, 4),
        time_seconds=round(time_seconds, 3),
        memory_mb=round(memory_mb, 1),
        findings_count=len(findings),
    )
