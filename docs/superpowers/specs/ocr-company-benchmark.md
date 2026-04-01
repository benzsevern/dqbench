# OCR Company Benchmark Spec

## Purpose

Benchmark tools that score OCR-extracted company names for review and suggest token-level corrections.

This is a post-OCR confidence and correction benchmark, not a generic OCR extraction benchmark.

## Dataset Shape

Each tier CSV contains:

- `record_id`
- `document_type`
- `company_name_truth`
- `company_name_ocr`
- `company_corrupted`

## Tiers

1. Tier 1: simple OCR substitutions and single-token drops
2. Tier 2: token-boundary drift such as merges and splits
3. Tier 3: adversarial suffix corruption, mixed OCR confusions, and layout-style token joins

## Adapter Contract

Implement `OCRCompanyAdapter.score_companies(csv_path) -> list[OCRCompanyPrediction]`

Each prediction row should return:

- `record_id`
- `confidence` in `[0, 1]`
- `weakest_token`
- `suggested_correction`
- optional `review_required`

## Metrics

- `confidence_separation`
- `clean_flag_rate`
- `corrupted_flag_rate`
- `weakest_token_hit_rate`
- `suggestion_coverage_rate`
- `suggestion_exact_hit_rate`
- `suggestion_improvement_rate`
- `avg_similarity_delta_on_suggestions`

## Composite Score

Per-tier composite:

- `25%` confidence separation
- `20%` clean non-flag rate
- `20%` corrupted flag rate
- `15%` weakest-token hit rate
- `10%` suggestion exact-hit rate
- `10%` suggestion improvement rate

Overall `DQBench OCR Company Score`:

- `Tier 1 * 20%`
- `Tier 2 * 40%`
- `Tier 3 * 40%`
