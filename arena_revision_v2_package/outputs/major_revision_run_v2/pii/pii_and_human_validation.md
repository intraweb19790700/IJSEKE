# PII and Human-Validation Status

- Run label: `CORRECTED_FAITHFUL_REIMPLEMENTATION` | Date: 2026-10-10
- **In-distribution pii_risk task metrics** (locked test, redacted view) are reported separately in `model/locked_test_metrics_by_seed.csv` (`pii_risk_macro_f1`).

## Configured synthetic PII stress test (separate from human validation)

Method: locked-test records whose original and redacted texts are identical (no configured PII) — the first 100 such records in file order — were injected with each configured synthetic PII type (phone `+1-555-XXXX`, email, `PID-XXXXXX`, SSN `XXX-XX-XXXX`; deterministic values). For each injected record: (a) redaction-rule recall = injected string removed by the local redaction rules; (b) student pii_risk head recall on the injected view (expected `contains_pii`); (c) student pii_risk head `none_or_redacted` rate on the redacted view. All 5 seeds (each model refit in memory).

This stress test uses ONLY configured synthetic PII and the model's own head; it is kept separate from an independent PII challenge set and is NOT human validation. Only the summary is saved (`synthetic_pii_stress_test_summary.csv`); no per-record stress-test file is persisted (storage constraint).

| pii_type | seed | support | rule recall | pii-head recall (injected) | pii-head none_or_redacted (redacted) |
|---|---:|---:|---:|---:|---:|
| email | 20260715 | 100 | 1.0000 | 0.8800 | 0.9900 |
| email | 20260716 | 100 | 1.0000 | 0.8800 | 0.9900 |
| email | 20260717 | 100 | 1.0000 | 0.8600 | 1.0000 |
| email | 20260718 | 100 | 1.0000 | 0.8600 | 0.9900 |
| email | 20260719 | 100 | 1.0000 | 0.9100 | 0.9900 |
| patient_id | 20260715 | 100 | 1.0000 | 0.0400 | 0.9700 |
| patient_id | 20260716 | 100 | 1.0000 | 0.0400 | 0.9900 |
| patient_id | 20260717 | 100 | 1.0000 | 0.0400 | 0.9800 |
| patient_id | 20260718 | 100 | 1.0000 | 0.0400 | 0.9800 |
| patient_id | 20260719 | 100 | 1.0000 | 0.0400 | 0.9900 |
| phone | 20260715 | 100 | 1.0000 | 0.0400 | 0.9900 |
| phone | 20260716 | 100 | 1.0000 | 0.0400 | 0.9900 |
| phone | 20260717 | 100 | 1.0000 | 0.0400 | 0.9900 |
| phone | 20260718 | 100 | 1.0000 | 0.0400 | 0.9900 |
| phone | 20260719 | 100 | 1.0000 | 0.0400 | 0.9900 |
| ssn | 20260715 | 100 | 1.0000 | 0.0600 | 0.9600 |
| ssn | 20260716 | 100 | 1.0000 | 0.0900 | 0.9700 |
| ssn | 20260717 | 100 | 1.0000 | 0.1000 | 0.9400 |
| ssn | 20260718 | 100 | 1.0000 | 0.1000 | 0.9500 |
| ssn | 20260719 | 100 | 1.0000 | 0.1000 | 0.9600 |
| ALL | 20260715 | 400 | 1.0000 | 0.2550 | 0.9775 |
| ALL | 20260716 | 400 | 1.0000 | 0.2625 | 0.9850 |
| ALL | 20260717 | 400 | 1.0000 | 0.2600 | 0.9775 |
| ALL | 20260718 | 400 | 1.0000 | 0.2600 | 0.9775 |
| ALL | 20260719 | 400 | 1.0000 | 0.2725 | 0.9825 |

## Independent PII challenge set / human validation

- Independent human PII labels: **ABSENT** -> **independent PII validation: BLOCKED**.
- Per-type precision/recall/F1/support against human labels and de-identified false negatives: **BLOCKED** (no human labels exist; none were fabricated).
- No agreement statistics are reported.
- Only an annotation template was created: `human_annotation_template.csv` (contains one clearly marked TEMPLATE EXAMPLE row, not a real annotation).

## LLM provenance (extracted only from real files/logs)

| Item | Status |
|---|---|
| LLM provider | ABSENT |
| LLM model/version | ABSENT |
| LLM dates | ABSENT |
| LLM generation settings | ABSENT |
| LLM prompts | ABSENT |
| LLM schema | ABSENT |
| LLM retries | ABSENT |
| adjudication logs | ABSENT |
| human annotations | ABSENT |
| human agreement statistics | BLOCKED |
| ethics records | ABSENT |

The dataset labels are LLM-assisted weak supervision as described in the manuscript, but no LLM provider/version/settings/prompts/logs, no independent human labels, and no ethics records are present in the package. See `llm_provenance_status.json`. Nothing was fabricated.
