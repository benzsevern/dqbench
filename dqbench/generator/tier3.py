"""Tier 3 dataset generator — 100,000-row healthcare claims with adversarial planted issues."""
from __future__ import annotations

import math
import random
from datetime import date, timedelta

import polars as pl

from dqbench.ground_truth import GroundTruth, PlantedColumn
from dqbench.generator.utils import FIRST_NAMES, LAST_NAMES

NROWS = 100_000

# ---------------------------------------------------------------------------
# Luhn helpers
# ---------------------------------------------------------------------------

def luhn_checksum(number_str: str) -> bool:
    """Return True if number_str passes the Luhn check."""
    digits = [int(d) for d in number_str]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits) + sum(sum(divmod(d * 2, 10)) for d in even_digits)
    return total % 10 == 0


def _make_valid_npi(rng: random.Random) -> str:
    """Generate a valid-looking 10-digit NPI that passes Luhn."""
    # Build 9 random digits then compute check digit
    while True:
        core = [rng.randint(0, 9) for _ in range(9)]
        # Compute check digit via Luhn
        # Sum without check digit
        all_digits = core + [0]
        odd_digits = all_digits[-1::-2]
        even_digits = all_digits[-2::-2]
        total = sum(odd_digits) + sum(sum(divmod(d * 2, 10)) for d in even_digits)
        check = (10 - (total % 10)) % 10
        candidate = "".join(str(d) for d in core) + str(check)
        if luhn_checksum(candidate):
            return candidate


def _corrupt_npi(npi: str, rng: random.Random) -> str:
    """Return a 10-digit string that looks like an NPI but fails Luhn."""
    digits = list(npi)
    # Flip the last digit by +1 (mod 10), which breaks the check digit
    last = int(digits[-1])
    bad_last = (last + rng.randint(1, 9)) % 10
    digits[-1] = str(bad_last)
    result = "".join(digits)
    # Verify it actually fails
    assert not luhn_checksum(result), f"Corruption failed for {npi} -> {result}"
    return result


# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

PROVIDER_TYPES = ["GP", "Specialist", "Hospital", "Clinic", "Lab"]
INSURANCE_PREFIXES = {
    "GP": "GP-",
    "Specialist": "SP-",
    "Hospital": "HO-",
    "Clinic": "CL-",
    "Lab": "LB-",
}
WRONG_PREFIXES = {
    "GP": ["SP-", "HO-", "CL-", "LB-"],
    "Specialist": ["GP-", "HO-", "CL-", "LB-"],
    "Hospital": ["GP-", "SP-", "CL-", "LB-"],
    "Clinic": ["GP-", "SP-", "HO-", "LB-"],
    "Lab": ["GP-", "SP-", "HO-", "CL-"],
}

# Valid ICD-10 format: letter + 2 digits + optional decimal + 1-2 chars
# e.g. J45.20, Z00.00, M79.3, K21.0
VALID_ICD10_CODES = [
    "J45.20", "Z00.00", "M79.3", "K21.0", "I10", "E11.9", "F32.1",
    "N39.0", "L30.9", "R05", "J06.9", "H52.4", "Z87.39", "M54.5",
    "G43.909", "K57.30", "J44.1", "D64.9", "E78.5", "B34.9",
]
# ICD-9 format codes (not valid ICD-10): 3-digit numeric + optional decimal
ICD9_CODES = ["250.00", "401.9", "272.4", "311", "428.0", "714.0", "493.90",
              "585.9", "250.01", "401.1", "272.0", "300.00", "414.01", "530.81"]

# Invalid ICD-10 (wrong format — numeric only, or too short)
INVALID_PROCEDURE_CODES = ["12345", "999", "AB", "00000", "ZZZZZ", "1A2",
                           "XY1", "000.00", "Z999", "99999", "A0", "B1C2D"]

FACILITY_CODES = ["FAC-001", "FAC-002", "FAC-003", "FAC-004", "FAC-005",
                  "FAC-006", "FAC-007", "FAC-008", "FAC-009", "FAC-010"]

BILLING_CODES = [f"BIL-{i:04d}" for i in range(1, 101)]
MODIFIER_CODES = ["25", "59", "GT", "95", "26", "TC", "50", "RT", "LT", "QW"]
PLACE_OF_SERVICE = ["11", "21", "22", "23", "24", "31", "32", "41", "42", "51"]
REVENUE_CODES = ["0100", "0200", "0250", "0270", "0300", "0360", "0370",
                 "0450", "0510", "0636"]
DRG_CODES = ["DRG-001", "DRG-002", "DRG-003", "DRG-004", "DRG-005"]
REFERRAL_SOURCES = ["PCP", "ER", "Self", "Specialist", "Urgent Care", "Telehealth"]
SERVICE_TYPES = ["Inpatient", "Outpatient", "Emergency", "Preventive", "Lab", "Imaging"]
CLAIM_STATUSES = ["Pending", "Approved", "Denied", "Under Review", "Paid", "Appealed"]
GENDERS = ["M", "F", "U"]

ATTENDING_PHYSICIANS = [
    f"Dr. {fn} {ln}"
    for fn, ln in [
        ("Alice", "Chen"), ("Bob", "Martinez"), ("Carol", "Johnson"),
        ("David", "Williams"), ("Eve", "Brown"), ("Frank", "Davis"),
        ("Grace", "Miller"), ("Henry", "Wilson"), ("Iris", "Moore"),
        ("James", "Taylor"), ("Karen", "Anderson"), ("Lee", "Thomas"),
    ]
]

PRIMARY_INSURANCES = [
    "BlueCross BlueShield", "Aetna", "Cigna", "UnitedHealth", "Humana",
    "Kaiser Permanente", "Anthem", "Centene", "Medicare", "Medicaid",
]
SECONDARY_INSURANCES = [None, None, None, None, "Medicare Supplement", "Medigap",
                         "Secondary Commercial", "Spouse Plan"]

CONFLICTING_DX_PAIRS: list[tuple[str, str]] = [
    # (primary, conflicting_secondary) — mutually exclusive conditions
    ("J45.20", "J44.1"),   # Asthma vs COPD — often mutually exclusive in strict coding
    ("E11.9", "E10.9"),    # T2DM vs T1DM
    ("I10", "I95.9"),      # Hypertension vs Hypotension
    ("F32.1", "F30.10"),   # Depression vs Mania
    ("K21.0", "K57.30"),   # GERD vs Diverticulitis (not conflicting per se, but used here)
    ("M79.3", "M79.3"),    # Same code — duplicate
    ("Z00.00", "Z00.01"),  # Two different wellness visit codes
    ("N39.0", "N40.0"),    # UTI vs BPH
    ("G43.909", "G43.019"), # Two migraine codes
    ("D64.9", "D50.9"),    # Anemia types
]

LATIN1_STRINGS = [
    "Dolor en el estómago",
    "Náuseas y vómitos",
    "Fiebre alta (38°C)",
    "Paciente con hipertensión",
    "Reacción alérgica — urticaria",
    "Diagnóstico: infección urinaria",
    "Dolor de cabeza severo",
    "Pérdida de apetito",
    "Evaluación de seguimiento",
    "Tratamiento completado con éxito",
    "Paciente estable",
    "Síntomas gastrointestinales",
    "Artritis reumatoide — dolor crónico",
    "Paciente anciano (75 años)",
    "Revisión post-quirúrgica",
]

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]

SMART_QUOTE_PHRASES = [
    "\u201cNormal\u201d BP reading",
    "Patient \u2018denies\u2019 chest pain",
    "\u201cStable\u201d post-op condition",
    "Referred to \u2018specialist\u2019",
    "\u201cImproved\u201d mobility after PT",
    "Lab values \u2018within range\u2019",
    "\u201cNo change\u201d in symptoms",
    "Provider noted \u2018mild edema\u2019",
    "\u201cFollowing up\u201d in 2 weeks",
    "Chart marked \u2018reviewed\u2019",
    "\u201cChronic condition\u201d managed",
    "Patient \u2018compliant\u2019 with meds",
    "\u201cAcute episode\u201d resolved",
    "Assessment: \u2018stable angina\u2019",
    "\u201cPre-existing\u201d condition noted",
    "Discharge: \u2018home with PT\u2019",
    "\u201cPost-surgical\u201d follow-up scheduled",
    "Notation: \u2018requires monitoring\u2019",
    "\u201cPending lab\u201d results",
    "Claim: \u2018approved pending review\u2019",
    "Note: \u2018patient educated\u2019",
    "Record: \u2018DNR on file\u2019",
    "\u201cAllergy\u201d documented in chart",
    "Provider: \u2018see attached\u2019",
    "Flag: \u2018high priority\u2019",
]

STRAIGHT_QUOTE_PHRASES = [
    '"Normal" BP reading',
    "Patient 'denies' chest pain",
    '"Stable" post-op condition',
    "Referred to 'specialist'",
    '"Improved" mobility after PT',
    "Lab values 'within range'",
    '"No change" in symptoms',
    "Provider noted 'mild edema'",
    '"Following up" in 2 weeks',
    "Chart marked 'reviewed'",
    '"Chronic condition" managed',
    "Patient 'compliant' with meds",
    '"Acute episode" resolved',
    "Assessment: 'stable angina'",
    '"Pre-existing" condition noted',
    "Discharge: 'home with PT'",
    '"Post-surgical" follow-up scheduled',
    "Notation: 'requires monitoring'",
    '"Pending lab" results',
    "Claim: 'approved pending review'",
    "Note: 'patient educated'",
    "Record: 'DNR on file'",
    '"Allergy" documented in chart',
    "Provider: 'see attached'",
    "Flag: 'high priority'",
]

PATIENT_PHONE_POOL = [
    "(555) 123-4567", "(555) 234-5678", "(555) 345-6789",
    "(555) 456-7890", "(555) 567-8901", "(555) 678-9012",
    "(555) 789-0123", "(555) 890-1234", "(555) 901-2345",
    "(555) 012-3456",
]

PRIOR_AUTH_FLAGS = ["Y", "N", "Pending", "Exempt"]

NUMERIC_PATIENT_NAMES = [
    "J0hn D03", "M4ry J4n3", "R0b3rt Sm1th", "P4tr1c14 D4v1s",
    "W1ll14m J0n3s", "J3nn1f3r G4rc14", "M1ch43l M1ll3r", "L1nd4 W1ls0n",
]

# Valid zip pattern but known-nonexistent ranges
NONEXISTENT_ZIPS = ["00000", "00001", "00002", "00003", "00004",
                    "99995", "99996", "99997", "99998", "99999"]

# US state abbreviations
STATE_ABBREVS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# Zip prefix → correct state (simplified mapping for mismatch detection)
ZIP_STATE_MAP: dict[str, str] = {
    "0": "CT", "1": "NY", "2": "VA", "3": "FL", "4": "OH",
    "5": "MN", "6": "IL", "7": "TX", "8": "CO", "9": "CA",
}
# States that would be wrong for each zip prefix
ZIP_STATE_WRONG: dict[str, list[str]] = {
    "0": ["TX", "FL", "CA", "WA", "MI"],
    "1": ["CA", "FL", "TX", "WA", "OR"],
    "2": ["TX", "CA", "FL", "WA", "OR"],
    "3": ["NY", "CA", "WA", "OR", "MI"],
    "4": ["CA", "FL", "TX", "WA", "OR"],
    "5": ["CA", "FL", "TX", "NY", "OR"],
    "6": ["CA", "FL", "TX", "WA", "NY"],
    "7": ["CA", "NY", "WA", "OR", "MI"],
    "8": ["FL", "TX", "NY", "OR", "MI"],
    "9": ["TX", "NY", "FL", "OH", "MI"],
}

# Weekend day of week check: 5=Saturday, 6=Sunday
WEEKDAY_ONLY_PROVIDERS = ["GP", "Specialist", "Clinic"]  # These don't work weekends


def _normal_sample(rng: random.Random, mean: float, std: float) -> float:
    """Box-Muller normal sample."""
    u1 = rng.random()
    u2 = rng.random()
    while u1 == 0.0:
        u1 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _next_weekday(d: date) -> date:
    """Return d if weekday, else advance to Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _to_weekend(d: date, rng: random.Random) -> date:
    """Shift d to the nearest upcoming Saturday or Sunday."""
    # Find next Saturday (5) or Sunday (6)
    days_ahead = (5 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    sat = d + timedelta(days=days_ahead)
    sun = sat + timedelta(days=1)
    return rng.choice([sat, sun])


def generate_tier3() -> tuple[pl.DataFrame, GroundTruth]:
    """Generate the Tier 3 benchmark dataset.

    Returns a 100,000-row healthcare claims DataFrame with 50 columns
    (25 planted issues, 25 clean) and the corresponding GroundTruth.
    """
    rng = random.Random(42)

    # ------------------------------------------------------------------ #
    # Pre-select issue rows                                               #
    # ------------------------------------------------------------------ #

    # 1. npi_number: 50 fail Luhn
    bad_npi_rows = rng.sample(range(NROWS), 50)
    bad_npi_set = set(bad_npi_rows)

    # 2. patient_state: 30 wrong for zip prefix
    wrong_state_rows = rng.sample(range(NROWS), 30)
    wrong_state_set = set(wrong_state_rows)

    # 3. service_date: 20 before date_of_birth
    service_before_dob_rows = rng.sample(range(NROWS), 20)
    service_before_dob_set = set(service_before_dob_rows)

    # 4. claim_notes: 15 Latin-1 chars
    latin1_note_rows = rng.sample(range(NROWS), 15)
    latin1_note_set = set(latin1_note_rows)

    # 5. provider_name: 10 zero-width Unicode chars
    zwsp_provider_rows = rng.sample(range(NROWS), 10)
    zwsp_provider_set = set(zwsp_provider_rows)

    # 6. diagnosis_desc: 25 smart quotes
    smart_quote_rows = rng.sample(range(NROWS), 25)
    smart_quote_set = set(smart_quote_rows)

    # 7. claim_amount vs policy_max_amount: 12 exceed policy max
    exceed_policy_rows = rng.sample(range(NROWS), 12)
    exceed_policy_set = set(exceed_policy_rows)

    # 8. service_day: 20 weekend dates for weekday-only providers
    weekend_service_rows = rng.sample(range(NROWS), 20)
    weekend_service_set = set(weekend_service_rows)

    # 9. submit_date: 15 before service_date
    submit_before_service_rows = rng.sample(range(NROWS), 15)
    submit_before_service_set = set(submit_before_service_rows)

    # 10. record_number: 8 gaps in sequential numbering — handled post-hoc
    gap_record_rows = sorted(rng.sample(range(1, NROWS), 8))  # indices to skip in seq

    # 11. patient_age: 5 don't match date_of_birth
    wrong_age_rows = rng.sample(range(NROWS), 5)
    wrong_age_set = set(wrong_age_rows)

    # 12. procedure_code: 18 invalid ICD-10 format
    bad_procedure_rows = rng.sample(range(NROWS), 18)
    bad_procedure_set = set(bad_procedure_rows)

    # 13. insurance_id: 22 wrong prefix for provider_type
    bad_insurance_rows = rng.sample(range(NROWS), 22)
    bad_insurance_set = set(bad_insurance_rows)

    # 14. patient_zip: 10 nonexistent zips (00000, 99999)
    nonexistent_zip_rows = rng.sample(range(NROWS), 10)
    nonexistent_zip_set = set(nonexistent_zip_rows)

    # 15. dosage_amount: 8 extreme outliers hidden in normal dist
    dosage_outlier_rows = rng.sample(range(NROWS), 8)
    dosage_outlier_set = set(dosage_outlier_rows)

    # 16. lab_result: 6 values in gap of bimodal distribution
    lab_gap_rows = rng.sample(range(NROWS), 6)
    lab_gap_set = set(lab_gap_rows)

    # 17/18. admission_date / discharge_date: 12 correlated (admission > discharge)
    admission_after_discharge_rows = set(rng.sample(range(NROWS), 12))

    # 19. primary_dx: 15 ICD-9 codes mixed in
    icd9_mix_rows = rng.sample(range(NROWS), 15)
    icd9_mix_set = set(icd9_mix_rows)

    # 20. secondary_dx: 10 conflict with primary_dx
    conflicting_dx_rows = rng.sample(
        [r for r in range(NROWS) if r not in icd9_mix_set], 10
    )
    conflicting_dx_set = set(conflicting_dx_rows)

    # 21. charge_code: 20 duplicates in sequential billing
    charge_dup_targets = rng.sample(range(NROWS), 20)
    charge_dup_sources = rng.sample(
        [r for r in range(NROWS) if r not in set(charge_dup_targets)], 20
    )

    # 22. patient_name: 8 numeric chars
    numeric_patient_rows = rng.sample(range(NROWS), 8)
    numeric_patient_set = set(numeric_patient_rows)
    numeric_patient_sorted = sorted(numeric_patient_rows)

    # 23. referring_npi: 25 fail Luhn
    bad_ref_npi_rows = rng.sample(range(NROWS), 25)
    bad_ref_npi_set = set(bad_ref_npi_rows)

    # 24. auth_number: 5 wrong length for insurance type
    bad_auth_rows = rng.sample(range(NROWS), 5)
    bad_auth_set = set(bad_auth_rows)

    # 25. payment_amount: 7 negative
    neg_payment_rows = rng.sample(range(NROWS), 7)
    neg_payment_set = set(neg_payment_rows)

    # ------------------------------------------------------------------ #
    # Generate base clean data first (date_of_birth, patient_zip, etc.)  #
    # ------------------------------------------------------------------ #

    dob_start = date(1930, 1, 1)
    dob_end = date(2005, 12, 31)
    service_start = date(2020, 1, 1)
    service_end = date(2024, 12, 31)

    # date_of_birth (clean)
    dobs: list[date] = [_rand_date(rng, dob_start, dob_end) for _ in range(NROWS)]

    # patient_zip (planted #14)
    # Valid range excludes the NONEXISTENT_ZIPS ranges (0-4 and 99995-99999)
    patient_zips: list[str] = []
    for i in range(NROWS):
        if i in nonexistent_zip_set:
            patient_zips.append(rng.choice(NONEXISTENT_ZIPS))
        else:
            # Generate zip in 10000-89999 to avoid overlap with nonexistent ranges
            patient_zips.append(f"{rng.randint(10000, 89999):05d}")

    # patient_state (planted #2 — wrong for zip prefix)
    patient_states: list[str] = []
    for i in range(NROWS):
        z = patient_zips[i]
        prefix = z[0]
        if i in wrong_state_set:
            wrong_opts = ZIP_STATE_WRONG.get(prefix, STATE_ABBREVS)
            patient_states.append(rng.choice(wrong_opts))
        else:
            correct_state = ZIP_STATE_MAP.get(prefix, rng.choice(STATE_ABBREVS))
            patient_states.append(correct_state)

    # provider_type (clean)
    provider_types: list[str] = [rng.choice(PROVIDER_TYPES) for _ in range(NROWS)]

    # service_date (planted #3 — 20 before dob)
    service_dates_raw: list[date] = []
    for i in range(NROWS):
        if i in service_before_dob_set:
            # before dob
            dob = dobs[i]
            service_dates_raw.append(dob - timedelta(days=rng.randint(1, 3650)))
        else:
            service_dates_raw.append(_rand_date(rng, service_start, service_end))

    # service_day (planted #8 — 20 weekend for weekday-only providers)
    service_days_raw: list[date] = []
    for i in range(NROWS):
        sd = service_dates_raw[i]
        if i in weekend_service_set and provider_types[i] in WEEKDAY_ONLY_PROVIDERS:
            service_days_raw.append(_to_weekend(sd, rng))
        else:
            # force to a weekday if provider is weekday-only
            if provider_types[i] in WEEKDAY_ONLY_PROVIDERS:
                service_days_raw.append(_next_weekday(sd))
            else:
                service_days_raw.append(sd)

    service_days: list[str] = [d.strftime("%Y-%m-%d") for d in service_days_raw]
    service_dates: list[str] = [d.strftime("%Y-%m-%d") for d in service_dates_raw]

    # submit_date (planted #9 — 15 before service_date)
    submit_dates: list[str] = []
    for i in range(NROWS):
        svc = service_dates_raw[i]
        if i in submit_before_service_set:
            sub = svc - timedelta(days=rng.randint(1, 30))
        else:
            sub = svc + timedelta(days=rng.randint(0, 45))
        submit_dates.append(sub.strftime("%Y-%m-%d"))

    # admission_date / discharge_date (planted #17/18)
    admission_dates: list[str] = []
    discharge_dates: list[str] = []
    for i in range(NROWS):
        svc = service_dates_raw[i]
        if i in admission_after_discharge_rows:
            # admission AFTER discharge
            disc = svc + timedelta(days=rng.randint(1, 5))
            adm = disc + timedelta(days=rng.randint(1, 3))
        else:
            adm = svc
            disc = svc + timedelta(days=rng.randint(0, 14))
        admission_dates.append(adm.strftime("%Y-%m-%d"))
        discharge_dates.append(disc.strftime("%Y-%m-%d"))

    # patient_age (planted #11 — 5 wrong)
    reference_date = date(2024, 1, 1)
    patient_ages: list[int] = []
    for i in range(NROWS):
        dob = dobs[i]
        correct_age = (reference_date - dob).days // 365
        if i in wrong_age_set:
            # Off by more than 5 years
            wrong_age = correct_age + rng.choice([-10, -8, -7, 10, 12, 15])
            patient_ages.append(max(1, wrong_age))
        else:
            patient_ages.append(correct_age)

    # npi_number (planted #1 — 50 fail Luhn)
    npi_numbers: list[str] = []
    for i in range(NROWS):
        valid_npi = _make_valid_npi(rng)
        if i in bad_npi_set:
            npi_numbers.append(_corrupt_npi(valid_npi, rng))
        else:
            npi_numbers.append(valid_npi)

    # referring_npi (planted #23 — 25 fail Luhn)
    referring_npis: list[str] = []
    for i in range(NROWS):
        valid_npi = _make_valid_npi(rng)
        if i in bad_ref_npi_set:
            referring_npis.append(_corrupt_npi(valid_npi, rng))
        else:
            referring_npis.append(valid_npi)

    # claim_notes (planted #4 — 15 Latin-1 chars)
    clean_claim_notes = [
        "Routine follow-up visit.",
        "Patient reports improvement.",
        "Labs ordered for monitoring.",
        "Prescription refilled.",
        "Referral to specialist placed.",
        "No acute distress observed.",
        "Patient education provided.",
        "Follow-up scheduled in 4 weeks.",
        "Imaging ordered.",
        "Vitals within normal range.",
    ]
    claim_notes: list[str] = []
    latin1_pool = LATIN1_STRINGS
    for i in range(NROWS):
        if i in latin1_note_set:
            claim_notes.append(rng.choice(latin1_pool))
        else:
            claim_notes.append(rng.choice(clean_claim_notes))

    # provider_name (planted #5 — 10 zero-width chars)
    base_provider_names = [
        "Dr. John Smith", "Dr. Mary Johnson", "Dr. Robert Brown",
        "Northwest Medical Center", "Eastside Clinic",
        "Dr. Jennifer Garcia", "Memorial Hospital",
        "Dr. Michael Miller", "City Health Lab",
        "Dr. Patricia Davis",
    ]
    provider_names: list[str] = []
    for i in range(NROWS):
        name = rng.choice(base_provider_names)
        if i in zwsp_provider_set:
            # Insert zero-width char in middle
            zwc = rng.choice(ZERO_WIDTH_CHARS)
            mid = len(name) // 2
            name = name[:mid] + zwc + name[mid:]
        provider_names.append(name)

    # diagnosis_desc (planted #6 — 25 smart quotes)
    # Build O(1) lookup: row -> phrase index for smart_quote rows
    smart_quote_row_idx: dict[int, int] = {
        r: j for j, r in enumerate(sorted(smart_quote_rows))
    }
    diagnosis_descs: list[str] = []
    for i in range(NROWS):
        if i in smart_quote_set:
            idx = smart_quote_row_idx[i]
            diagnosis_descs.append(SMART_QUOTE_PHRASES[idx % len(SMART_QUOTE_PHRASES)])
        else:
            diagnosis_descs.append(rng.choice(STRAIGHT_QUOTE_PHRASES))

    # policy_max_amount (clean)
    policy_max_amounts: list[float] = []
    for _ in range(NROWS):
        policy_max_amounts.append(round(rng.choice([5000.0, 10000.0, 25000.0, 50000.0, 100000.0]), 2))

    # claim_amount (planted #7 — 12 exceed policy_max)
    claim_amounts: list[float] = []
    for i in range(NROWS):
        if i in exceed_policy_set:
            # Exceed by 10-50%
            factor = 1.1 + rng.random() * 0.4
            claim_amounts.append(round(policy_max_amounts[i] * factor, 2))
        else:
            claim_amounts.append(round(rng.uniform(50.0, policy_max_amounts[i] * 0.95), 2))

    # record_number (planted #10 — 8 gaps)
    # Sequential 1..NROWS but with 8 numbers skipped (creating gaps)
    gap_set = set(gap_record_rows)
    record_numbers: list[int] = []
    seq = 1
    for i in range(NROWS):
        if i in gap_set:
            seq += 2  # skip one number → gap
        record_numbers.append(seq)
        seq += 1

    # procedure_code (planted #12 — 18 invalid ICD-10)
    procedure_codes: list[str] = []
    for i in range(NROWS):
        if i in bad_procedure_set:
            procedure_codes.append(rng.choice(INVALID_PROCEDURE_CODES))
        else:
            procedure_codes.append(rng.choice(VALID_ICD10_CODES))

    # insurance_id (planted #13 — 22 wrong prefix)
    insurance_ids: list[str] = []
    for i in range(NROWS):
        ptype = provider_types[i]
        if i in bad_insurance_set:
            wrong_prefix = rng.choice(WRONG_PREFIXES[ptype])
            suffix = format(rng.randint(100000, 999999), "06d")
            insurance_ids.append(f"{wrong_prefix}{suffix}")
        else:
            correct_prefix = INSURANCE_PREFIXES[ptype]
            suffix = format(rng.randint(100000, 999999), "06d")
            insurance_ids.append(f"{correct_prefix}{suffix}")

    # dosage_amount (planted #15 — 8 extreme outliers in normal dist)
    DOSAGE_MEAN = 250.0
    DOSAGE_STD = 50.0
    dosage_amounts: list[float] = []
    for i in range(NROWS):
        if i in dosage_outlier_set:
            # Hidden extreme: 4-6 stddev above mean
            val = DOSAGE_MEAN + (4.0 + rng.random() * 2.0) * DOSAGE_STD
        else:
            val = max(1.0, _normal_sample(rng, DOSAGE_MEAN, DOSAGE_STD))
        dosage_amounts.append(round(val, 2))

    # lab_result (planted #16 — 6 values in gap of bimodal distribution)
    # Bimodal: cluster A around 20 (clamped <=38), cluster B around 80 (clamped >=62)
    # Gap is 39-61; planted values land in 42-58
    lab_results: list[float] = []
    for i in range(NROWS):
        if i in lab_gap_set:
            # Place in gap: 42-58
            lab_results.append(round(rng.uniform(42.0, 58.0), 2))
        else:
            if rng.random() < 0.5:
                # Cluster A: ~20, clamped to [0, 38] to stay out of gap
                val = _normal_sample(rng, 20.0, 5.0)
                lab_results.append(round(min(38.0, max(0.0, val)), 2))
            else:
                # Cluster B: ~80, clamped to [62, 120] to stay out of gap
                val = _normal_sample(rng, 80.0, 8.0)
                lab_results.append(round(max(62.0, min(120.0, val)), 2))

    # primary_dx (planted #19 — 15 ICD-9 mixed)
    primary_dxs: list[str] = []
    for i in range(NROWS):
        if i in icd9_mix_set:
            primary_dxs.append(rng.choice(ICD9_CODES))
        else:
            primary_dxs.append(rng.choice(VALID_ICD10_CODES))

    # secondary_dx (planted #20 — 10 conflict with primary)
    secondary_dxs: list[str] = []
    for i in range(NROWS):
        if i in conflicting_dx_set:
            # Pick a conflicting pair based on primary_dx
            pdx = primary_dxs[i]
            # Find a conflicting secondary
            pair = rng.choice(CONFLICTING_DX_PAIRS)
            if pair[0] == pdx:
                secondary_dxs.append(pair[1])
            else:
                secondary_dxs.append(pair[0])
        else:
            secondary_dxs.append(rng.choice(VALID_ICD10_CODES + [None, None]))  # type: ignore[arg-type]

    # charge_code (planted #21 — 20 duplicates)
    charge_codes_base: list[str] = []
    seen_cc: set[str] = set()
    cc_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for _ in range(NROWS):
        while True:
            cc = "CHG-" + "".join(rng.choice(cc_chars) for _ in range(6))
            if cc not in seen_cc:
                seen_cc.add(cc)
                charge_codes_base.append(cc)
                break
    for target, source in zip(charge_dup_targets, charge_dup_sources):
        charge_codes_base[target] = charge_codes_base[source]

    # patient_name (planted #22 — 8 numeric)
    patient_names: list[str] = []
    for i in range(NROWS):
        if i in numeric_patient_set:
            idx = numeric_patient_sorted.index(i)
            patient_names.append(NUMERIC_PATIENT_NAMES[idx % len(NUMERIC_PATIENT_NAMES)])
        else:
            fn = rng.choice(FIRST_NAMES)
            ln = rng.choice(LAST_NAMES)
            patient_names.append(f"{fn} {ln}")

    # auth_number (planted #24 — 5 wrong length)
    # Standard auth: 10 digits; wrong: 6 or 15 digits
    auth_numbers: list[str] = []
    for i in range(NROWS):
        if i in bad_auth_set:
            bad_len = rng.choice([6, 15])
            auth_numbers.append("".join(str(rng.randint(0, 9)) for _ in range(bad_len)))
        else:
            auth_numbers.append("".join(str(rng.randint(0, 9)) for _ in range(10)))

    # payment_amount (planted #25 — 7 negative)
    payment_amounts: list[float] = []
    for i in range(NROWS):
        if i in neg_payment_set:
            payment_amounts.append(round(-rng.uniform(10.0, 500.0), 2))
        else:
            payment_amounts.append(round(rng.uniform(0.0, claim_amounts[i]), 2))

    # ------------------------------------------------------------------ #
    # Clean columns (25 total)                                           #
    # ------------------------------------------------------------------ #

    # patient_id
    patient_ids: list[str] = [f"PAT-{i+1:07d}" for i in range(NROWS)]

    # gender
    genders: list[str] = [rng.choice(GENDERS) for _ in range(NROWS)]

    # facility_code
    facility_codes: list[str] = [rng.choice(FACILITY_CODES) for _ in range(NROWS)]

    # billing_code
    billing_codes_col: list[str] = [rng.choice(BILLING_CODES) for _ in range(NROWS)]

    # modifier_code — 40% null (optional)
    modifier_codes: list[str | None] = []
    for _ in range(NROWS):
        modifier_codes.append(None if rng.random() < 0.4 else rng.choice(MODIFIER_CODES))

    # place_of_service
    pos_col: list[str] = [rng.choice(PLACE_OF_SERVICE) for _ in range(NROWS)]

    # revenue_code
    revenue_codes_col: list[str] = [rng.choice(REVENUE_CODES) for _ in range(NROWS)]

    # drg_code — 50% null (applies to inpatient only)
    drg_codes: list[str | None] = []
    for _ in range(NROWS):
        drg_codes.append(None if rng.random() < 0.5 else rng.choice(DRG_CODES))

    # attending_physician
    attending_physicians: list[str] = [rng.choice(ATTENDING_PHYSICIANS) for _ in range(NROWS)]

    # referral_source
    referral_sources: list[str] = [rng.choice(REFERRAL_SOURCES) for _ in range(NROWS)]

    # patient_phone
    patient_phones: list[str] = []
    for _ in range(NROWS):
        area = rng.randint(200, 999)
        prefix = rng.randint(200, 999)
        line = rng.randint(1000, 9999)
        patient_phones.append(f"({area}) {prefix}-{line}")

    # patient_email
    patient_emails: list[str] = []
    for _ in range(NROWS):
        fn = rng.choice(FIRST_NAMES).lower()
        ln = rng.choice(LAST_NAMES).lower()
        domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "healthmail.org"])
        patient_emails.append(f"{fn}.{ln}@{domain}")

    # emergency_contact — 30% null
    emergency_contacts: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.3:
            emergency_contacts.append(None)
        else:
            fn = rng.choice(FIRST_NAMES)
            ln = rng.choice(LAST_NAMES)
            emergency_contacts.append(f"{fn} {ln}")

    # primary_insurance
    primary_insurances: list[str] = [rng.choice(PRIMARY_INSURANCES) for _ in range(NROWS)]

    # secondary_insurance — 70% null
    secondary_insurances: list[str | None] = []
    for _ in range(NROWS):
        if rng.random() < 0.7:
            secondary_insurances.append(None)
        else:
            secondary_insurances.append(rng.choice([s for s in SECONDARY_INSURANCES if s is not None]))

    # copay_amount — valid
    copay_amounts: list[float] = [
        round(rng.choice([0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0]), 2)
        for _ in range(NROWS)
    ]

    # deductible_met — boolean string
    deductible_met: list[str] = [rng.choice(["Y", "N"]) for _ in range(NROWS)]

    # prior_auth_flag
    prior_auth_flags: list[str] = [rng.choice(PRIOR_AUTH_FLAGS) for _ in range(NROWS)]

    # service_type
    service_types: list[str] = [rng.choice(SERVICE_TYPES) for _ in range(NROWS)]

    # claim_status
    claim_statuses: list[str] = [rng.choice(CLAIM_STATUSES) for _ in range(NROWS)]

    # adjudication_date — after submit_date, 20% null (still pending)
    adjudication_dates: list[str | None] = []
    for i in range(NROWS):
        if rng.random() < 0.2:
            adjudication_dates.append(None)
        else:
            sub_date = date.fromisoformat(submit_dates[i])
            adj_date = sub_date + timedelta(days=rng.randint(1, 60))
            adjudication_dates.append(adj_date.strftime("%Y-%m-%d"))

    # remittance_amount — close to payment_amount, valid
    remittance_amounts: list[float] = []
    for i in range(NROWS):
        pa = payment_amounts[i]
        if pa < 0:
            # For negative payments (planted issue), remittance is 0
            remittance_amounts.append(0.0)
        else:
            remittance_amounts.append(round(pa * rng.uniform(0.95, 1.0), 2))

    # dob as string
    dobs_str: list[str] = [d.strftime("%Y-%m-%d") for d in dobs]

    # ------------------------------------------------------------------ #
    # Build Polars DataFrame (50 columns)                                #
    # ------------------------------------------------------------------ #

    # secondary_dx may contain None — keep as Utf8
    secondary_dxs_clean: list[str | None] = [
        v if isinstance(v, str) else None for v in secondary_dxs
    ]

    df = pl.DataFrame(
        {
            # Planted (25)
            "npi_number":       npi_numbers,
            "patient_state":    patient_states,
            "service_date":     service_dates,
            "claim_notes":      claim_notes,
            "provider_name":    provider_names,
            "diagnosis_desc":   diagnosis_descs,
            "claim_amount":     claim_amounts,
            "service_day":      service_days,
            "submit_date":      submit_dates,
            "record_number":    record_numbers,
            "patient_age":      patient_ages,
            "procedure_code":   procedure_codes,
            "insurance_id":     insurance_ids,
            "patient_zip":      patient_zips,
            "dosage_amount":    dosage_amounts,
            "lab_result":       lab_results,
            "admission_date":   admission_dates,
            "discharge_date":   discharge_dates,
            "primary_dx":       primary_dxs,
            "secondary_dx":     secondary_dxs_clean,
            "charge_code":      charge_codes_base,
            "patient_name":     patient_names,
            "referring_npi":    referring_npis,
            "auth_number":      auth_numbers,
            "payment_amount":   payment_amounts,
            # Clean (25)
            "patient_id":           patient_ids,
            "date_of_birth":        dobs_str,
            "gender":               genders,
            "policy_max_amount":    policy_max_amounts,
            "provider_type":        provider_types,
            "facility_code":        facility_codes,
            "billing_code":         billing_codes_col,
            "modifier_code":        modifier_codes,
            "place_of_service":     pos_col,
            "revenue_code":         revenue_codes_col,
            "drg_code":             drg_codes,
            "attending_physician":  attending_physicians,
            "referral_source":      referral_sources,
            "patient_phone":        patient_phones,
            "patient_email":        patient_emails,
            "emergency_contact":    emergency_contacts,
            "primary_insurance":    primary_insurances,
            "secondary_insurance":  secondary_insurances,
            "copay_amount":         copay_amounts,
            "deductible_met":       deductible_met,
            "prior_auth_flag":      prior_auth_flags,
            "service_type":         service_types,
            "claim_status":         claim_statuses,
            "adjudication_date":    adjudication_dates,
            "remittance_amount":    remittance_amounts,
        }
    )

    # ------------------------------------------------------------------ #
    # Build GroundTruth                                                   #
    # ------------------------------------------------------------------ #

    planted: dict[str, PlantedColumn] = {
        "npi_number": PlantedColumn(
            issues=["luhn"],
            planted_count=50,
            description="50 NPI numbers are valid-looking 10-digit strings that fail the Luhn check.",
            affected_rows=sorted(bad_npi_rows),
        ),
        "patient_state": PlantedColumn(
            issues=["cross_column"],
            planted_count=30,
            description="30 rows have a valid state abbreviation that is inconsistent with the patient_zip prefix.",
            affected_rows=sorted(wrong_state_rows),
        ),
        "service_date": PlantedColumn(
            issues=["logic_violation"],
            planted_count=20,
            description="20 rows have service_date before the patient's date_of_birth.",
            affected_rows=sorted(service_before_dob_rows),
        ),
        "claim_notes": PlantedColumn(
            issues=["encoding"],
            planted_count=15,
            description="15 rows contain Latin-1 characters (é, ñ, ü) that cause UTF-8 encoding issues.",
            affected_rows=sorted(latin1_note_rows),
        ),
        "provider_name": PlantedColumn(
            issues=["encoding"],
            planted_count=10,
            description="10 rows contain zero-width Unicode characters embedded in provider names.",
            affected_rows=sorted(zwsp_provider_rows),
        ),
        "diagnosis_desc": PlantedColumn(
            issues=["encoding"],
            planted_count=25,
            description="25 rows use smart/curly quotes instead of straight ASCII quotes.",
            affected_rows=sorted(smart_quote_rows),
        ),
        "claim_amount": PlantedColumn(
            issues=["logic_violation"],
            planted_count=12,
            description="12 rows have claim_amount exceeding the patient's policy_max_amount.",
            affected_rows=sorted(exceed_policy_rows),
        ),
        "service_day": PlantedColumn(
            issues=["semantic"],
            planted_count=20,
            description="20 rows have weekend service dates for providers that only operate on weekdays.",
            affected_rows=sorted(weekend_service_rows),
        ),
        "submit_date": PlantedColumn(
            issues=["logic_violation"],
            planted_count=15,
            description="15 rows have submit_date before service_date.",
            affected_rows=sorted(submit_before_service_rows),
        ),
        "record_number": PlantedColumn(
            issues=["sequence_gap"],
            planted_count=8,
            description="8 gaps in the sequential record_number series (numbers skipped in billing sequence).",
            affected_rows=sorted(gap_record_rows),
        ),
        "patient_age": PlantedColumn(
            issues=["logic_violation"],
            planted_count=5,
            description="5 rows have patient_age that does not match the calculated age from date_of_birth.",
            affected_rows=sorted(wrong_age_rows),
        ),
        "procedure_code": PlantedColumn(
            issues=["invalid_format"],
            planted_count=18,
            description="18 rows have procedure codes in invalid ICD-10 format (numeric only, too short, etc.).",
            affected_rows=sorted(bad_procedure_rows),
        ),
        "insurance_id": PlantedColumn(
            issues=["logic_violation"],
            planted_count=22,
            description="22 rows have insurance_id with a prefix inconsistent with the provider_type.",
            affected_rows=sorted(bad_insurance_rows),
        ),
        "patient_zip": PlantedColumn(
            issues=["invalid_values"],
            planted_count=10,
            description="10 rows have valid-format zip codes (00000, 99999) that correspond to nonexistent zip ranges.",
            affected_rows=sorted(nonexistent_zip_rows),
        ),
        "dosage_amount": PlantedColumn(
            issues=["outlier_values"],
            planted_count=8,
            description="8 rows have extreme dosage outliers (4-6 stddev above mean) hidden within a normal distribution.",
            affected_rows=sorted(dosage_outlier_rows),
        ),
        "lab_result": PlantedColumn(
            issues=["distribution_anomaly"],
            planted_count=6,
            description="6 rows have lab_result values in the gap (42-58) of a bimodal distribution (cluster A clamped <=38, cluster B clamped >=62).",
            affected_rows=sorted(lab_gap_rows),
        ),
        "admission_date": PlantedColumn(
            issues=["logic_violation"],
            planted_count=12,
            description="12 rows have admission_date after discharge_date (correlated issue).",
            affected_rows=sorted(admission_after_discharge_rows),
        ),
        "discharge_date": PlantedColumn(
            issues=["logic_violation"],
            planted_count=12,
            description="12 rows have discharge_date before admission_date (correlated with admission_date issue).",
            affected_rows=sorted(admission_after_discharge_rows),
        ),
        "primary_dx": PlantedColumn(
            issues=["mixed_coding_standard"],
            planted_count=15,
            description="15 rows have ICD-9 format codes mixed into an ICD-10 coded dataset.",
            affected_rows=sorted(icd9_mix_rows),
        ),
        "secondary_dx": PlantedColumn(
            issues=["logic_violation"],
            planted_count=10,
            description="10 rows have secondary_dx that conflicts with (is mutually exclusive of) primary_dx.",
            affected_rows=sorted(conflicting_dx_rows),
        ),
        "charge_code": PlantedColumn(
            issues=["duplicate_values"],
            planted_count=20,
            description="20 rows have duplicate charge codes in what should be a sequential unique billing series.",
            affected_rows=sorted(charge_dup_targets),
        ),
        "patient_name": PlantedColumn(
            issues=["wrong_type"],
            planted_count=8,
            description="8 rows have numeric characters substituted into patient names (e.g. 'J0hn D03').",
            affected_rows=sorted(numeric_patient_rows),
        ),
        "referring_npi": PlantedColumn(
            issues=["luhn"],
            planted_count=25,
            description="25 referring NPI numbers fail the Luhn check.",
            affected_rows=sorted(bad_ref_npi_rows),
        ),
        "auth_number": PlantedColumn(
            issues=["invalid_format"],
            planted_count=5,
            description="5 rows have auth_number with wrong length (should be 10 digits; has 6 or 15).",
            affected_rows=sorted(bad_auth_rows),
        ),
        "payment_amount": PlantedColumn(
            issues=["invalid_values"],
            planted_count=7,
            description="7 rows have negative payment_amount (refunds mislabeled as negative payments).",
            affected_rows=sorted(neg_payment_rows),
        ),
    }

    clean = [
        "patient_id", "date_of_birth", "gender", "policy_max_amount",
        "provider_type", "facility_code", "billing_code", "modifier_code",
        "place_of_service", "revenue_code", "drg_code", "attending_physician",
        "referral_source", "patient_phone", "patient_email", "emergency_contact",
        "primary_insurance", "secondary_insurance", "copay_amount", "deductible_met",
        "prior_auth_flag", "service_type", "claim_status", "adjudication_date",
        "remittance_amount",
    ]

    total = sum(p.planted_count for p in planted.values())

    gt = GroundTruth(
        tier=3,
        version="1.0.0",
        rows=NROWS,
        columns=50,
        planted_columns=planted,
        clean_columns=clean,
        total_planted_issues=total,
    )

    return df, gt
