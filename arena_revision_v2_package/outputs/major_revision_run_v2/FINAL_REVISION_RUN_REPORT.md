# FINAL REVISION RUN REPORT — RC-EdgeDistill (IJSEKE-D-26-00185)

- **Run label: `CORRECTED_FAITHFUL_REIMPLEMENTATION`** — this is a **corrected faithful reimplementation**, **NOT an exact reproduction**. The exact submitted implementation is not available in the workspace; the manuscript Section 3.3 specification was re-implemented from scratch in pure numpy/scipy/scikit-learn, with the reference `code_reference/student.py` symmetric-KL backward pass **corrected** and the reference 40-epoch/mini-batch schedule **replaced by the manuscript's 160 full-batch epochs**. See `implementation_assumptions.md`.
- Date: 2026-10-10 | Repository: `/home/user/IJSEKE` | Run root: `/home/user/IJSEKE/arena_revision_v2_package/outputs/major_revision_run_v2`
- Hardware/software: CPU-only (2 cores), no GPU, `OMP_NUM_THREADS=1`, Python 3.11.2, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1
- The submitted reference metrics (**0.6896 composite macro-F1 / 0.9867 consistency / 0.2042 ms per record**) are **NOT claimed as reproduced**: no original checkpoints, predictions, or logs exist in the package to verify them. All numbers below come from this run on a NEW group-aware locked split.

## 1. Commands

See `reproducibility/run_commands.txt` for the exact stage commands. Summary:

```bash
cd arena_revision_v2_package/outputs/major_revision_run_v2/src
OMP_NUM_THREADS=1 python3 stage1_input_audit.py      # input check + leakage audit + locked split
OMP_NUM_THREADS=1 python3 gradient_check.py          # finite-difference check of corrected symmetric KL
OMP_NUM_THREADS=1 python3 stage2_train_student.py --epochs 160   # lambda grid x 5 seeds (validation only)
OMP_NUM_THREADS=1 python3 stage3_test_student.py     # retrain per seed, locked test once per seed
OMP_NUM_THREADS=1 python3 stage4_baselines.py        # TF-IDF baselines (validation-selected)
OMP_NUM_THREADS=1 python3 stage5_timing.py           # common end-to-end timing
OMP_NUM_THREADS=1 python3 stage6_pii.py              # synthetic PII stress + validation status
OMP_NUM_THREADS=1 python3 stage7_repro_report.py     # this report + reproducibility package
```

## 2. Input verification, data and hashes

- Dataset: `arena_revision_v2_package/input/prepared_paired_dataset_2000.csv` — **2000 records** (expected 2,000: PASS).
- **Data SHA-256:** `bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e` — matches the expected `bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e`: **True**.
- Manuscript, decision letter, and revision plan readable: PASS; all 12 required `code_reference/*.py` files present: PASS (`data/input_verification.json`).
- **Code SHA-256** (src/*.py of this run): `fdc9a792c713446b27ffe746d5687ad39d0967425865ce377cb2b0d3a5817110`
- **Split seed:** 20260720 (fixed, recorded); model seeds: [20260715, 20260716, 20260717, 20260718, 20260719]
- Data columns: `record_id, language, source_family, split, original_text, redacted_text, service_domain, sentiment, priority, pii_original, pii_redacted, confidence`

## 3. Gradient check (corrected symmetric-KL backward pass) — REQUIRED GATE

The reference `code_reference/student.py` ships an incomplete symmetric-KL backward pass (`lam*0.5*(p_o-p_r)` / `lam*0.5*(p_r-p_o)`), which is not the gradient of `0.5*(KL(p||q)+KL(q||p))`. This run implements the complete batch-mean logit gradients with epsilon clipping:

```text
g_p = 0.5/B * [ p*(log(p) - log(q) - KL(p||q)) + (p - q) ]
g_q = 0.5/B * [ q*(log(q) - log(p) - KL(q||p)) + (q - p) ]
```

Finite-difference verification (central differences, h=1e-6, tiny dense fixtures; details in `gradient_check/`):

- Symmetric-KL logit gradients: max abs error **3.036e-10** vs tolerance 1.0e-06 -> **PASS**
- Full student parameter gradients (CE + corrected KL + shared layer + heads): max abs error **2.949e-09** vs tolerance 1.0e-05 -> **PASS**
- **Overall: PASS** (training did not proceed unless PASS)

## 4. Split audit (leakage-safe, group-aware locked split)

- Rows: 2000; locked split counts: train 1400, validation 301, test 299 (targets 70/15/15; group-aware greedy assignment, seed 20260720).
- Exact duplicates (NFKC + casefold + whitespace collapse, both views): **120 pairs**, 85 member rows.
- Near duplicates (char_wb (3,5) TF-IDF cosine >= 0.9, both views): **448 pairs**, 383 member rows.
- Lineage groups (union-find): **1755** (largest group 13).
- **Crossing counts:** groups crossing splits = **0**, exact-duplicate pairs separated = **0**, near-duplicate pairs separated = **0** (all must be 0).
- Metadata: hospital / website / account / collection batch / synthetic-parent id = **ABSENT** (columns not present); `pii_original` used as synthetic-injection proxy. Source-level grouping beyond duplicate lineage is therefore not possible — recorded limitation.
- Language/source confounding: **perfectly confounded** (en <-> english_public_hospital_reviews, zh <-> chinese_public_web_deidentified); subgroup results are a **joint language/source diagnostic** — no separate causal language or source effects are claimed.
- Text views: 1154 rows changed by redaction, 846 unchanged.
- Full details: `data/input_verification.{json,md}`, `data/hashes.json`, `data/leakage_audit.{csv,json,md}`, `data/locked_split.csv`, `data/locked_split_manifest.json`.

## 5. Model selection (validation only; preregistered rule)

Rule (declared in `model/selection_config.yaml` **before** any training, and applied before any test prediction): maximize validation composite macro-F1; within tolerance 0.005 maximize validation mean consistency; then choose the smaller lambda. Locked-test labels/predictions were never used for selection.

Training: 160 FULL-BATCH epochs, Adam lr=0.006, weight decay 1e-4, dropout 0.15, shared 384, seeds 20260715–20260719, lambda grid {0, 0.03, 0.05, 0.10, 0.25}.

| lambda | val composite macro-F1 (mean) | val mean consistency (mean) |
|---:|---:|---:|
| 0.0 | 0.5495 | 0.9615 |
| 0.03 | 0.5503 | 0.9615 |
| 0.05 | 0.5505 | 0.9619 |
| 0.1 | 0.5503 | 0.9628 |
| 0.25 | 0.5504 | 0.9643 | **selected**

**Selected lambda: 0.25** (best validation composite macro-F1 0.550492).

## 6. Locked-test metrics — RC-EdgeDistill (selected lambda, 5 seeds, once each)

Each seed was retrained in memory (no checkpoints) and the locked test was evaluated exactly once per seed. Primary task metrics on the redacted view (inference convention); composite = mean of the three semantic-task macro-F1 values.

| seed | composite | service_domain | sentiment | priority | pii_risk | consistency (all) | consistency (changed) | consistency (unchanged) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260715 | 0.5294 | 0.4133 | 0.5425 | 0.6325 | 0.4975 | 0.9721 | 0.9462 | 1.0000 |
| 20260716 | 0.5432 | 0.4382 | 0.5568 | 0.6346 | 0.4975 | 0.9666 | 0.9355 | 1.0000 |
| 20260717 | 0.5498 | 0.4354 | 0.5715 | 0.6423 | 0.4983 | 0.9666 | 0.9355 | 1.0000 |
| 20260718 | 0.5419 | 0.4358 | 0.5564 | 0.6337 | 0.4983 | 0.9632 | 0.9290 | 1.0000 |
| 20260719 | 0.5462 | 0.4405 | 0.5567 | 0.6412 | 0.4975 | 0.9710 | 0.9441 | 1.0000 |
| **mean** | **0.5421** | **0.4326** | **0.5568** | **0.6369** | **0.4978** | **0.9679** | **0.9381** | **1.0000** |
| **std** | 0.0077 | 0.0110 | 0.0103 | 0.0045 | 0.0005 | 0.0036 | 0.0070 | 0.0000 |

- Locked test rows: 299 (155 changed by redaction, 144 unchanged).
- Original-view composite macro-F1 (mean): 0.5379.
- Paired bootstrap 95% CIs (2,000 replicates, seed 20260721, percentile; mean across seeds):
  - composite_macro_f1: [0.5011, 0.5757]
  - mean_consistency_all: [0.9552, 0.9797]
  - mean_consistency_changed: [0.9145, 0.9601]
  - pii_risk_macro_f1: [0.4948, 0.5000]
  - priority_macro_f1: [0.5768, 0.6928]
  - sentiment_macro_f1: [0.5009, 0.6099]
  - service_domain_macro_f1: [0.3495, 0.5068]
- Per-class precision/recall/F1/support: `model/per_class_metrics.csv`. ONE combined per-record prediction CSV (all seeds): `model/locked_test_predictions.csv`.

### Subgroup (joint language/source diagnostic — perfectly confounded)

| subgroup | n | composite macro-F1 (mean) | mean consistency (mean) | pii_risk F1 (mean) |
|---|---:|---:|---:|---:|
| en / english_public_hospital_reviews | 142 | 0.5747 | 0.9939 | 0.4965 |
| zh / chinese_public_web_deidentified | 157 | 0.4001 | 0.9444 | 0.4990 |

Language and source_family are the same partition (perfectly confounded): these are **joint** diagnostics, not separable language or source effects. Any performance difference between the two subgroups cannot be attributed to language alone — the Chinese records are public-web excerpts and the English records are hospital reviews from a different country/source, so **language and source are confounded** in this dataset.

### Rare class: medication (service_domain)

- Locked-test support: **4** records; F1 0.3333 (precision 0.5000, recall 0.2500). Support is very small — treat as unstable.

## 7. Baselines (same locked split; parameters selected on validation only)

| baseline | selected param | test composite macro-F1 (mean) | std across seeds | consistency (all) | en | zh | medication F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_complement_nb | alpha=0.5 | 0.4064 | 0.0000 | 0.9175 | 0.2365 | 0.2647 | 0.3333 |
| tfidf_linear_svc_balanced | C=1.0 | 0.5648 | 0.0000 | 0.9342 | 0.6019 | 0.4061 | 0.3333 |
| tfidf_logistic_balanced | C=4.0 | 0.5636 | 0.0000 | 0.9420 | 0.5859 | 0.3967 | 0.3333 |
| tfidf_ridge_balanced | alpha=1.0 | 0.5663 | 0.0000 | 0.9365 | 0.5858 | 0.4034 | 0.3333 |
| tfidf_sgd_modified_huber | alpha=0.001 | 0.5591 | 0.0046 | 0.9338 | 0.5913 | 0.4013 | 0.3333 |
| **rc_edgedistill (lambda 0.25)** | — | **0.5421** | **0.0077** | **0.9679** | **0.5747** | **0.4001** | **0.3333** |

- Details: `baselines/hyperparameters.csv` (selection), `baselines/validation_results.csv`, `baselines/locked_test_results.csv`, `baselines/per_class_results.csv`, `baselines/bootstrap_confidence_intervals.csv`, `baselines/locked_test_summary_across_seeds.csv`. No per-baseline prediction files and no `fitted_models.pkl` are saved (storage constraint).
- **Compact multilingual encoder baseline: BLOCKED** — torch/sentence-transformers are not installed in this CPU-only environment; no substitute architecture was used (`baselines/encoder_baseline_status.json`).
- **Frozen sentence-embedding baselines (MiniLM / multilingual-E5 / BGE-small): not runnable here** — sentence-transformers is not installed; the submitted embedding numbers could not be verified and are not reproduced.

## 8. Common end-to-end timing (CPU, threads=1, per-record, 1 warm-up + 5 repetitions)

Pipeline per record (identical boundary for every method): input normalization (NFKC + whitespace collapse) -> local redaction (regex) -> TF-IDF construction -> inference -> decoding -> JSON serialization. Model-only latency is not mixed in.

| method | mean ms/record | p50 | p95 | throughput (rec/s) |
|---|---:|---:|---:|---:|
| rc_edgedistill_lambda_0.25 | 1.526 | 1.453 | 2.038 | 655.2 |
| tfidf_logistic_balanced | 4.304 | 4.127 | 5.654 | 232.3 |
| tfidf_linear_svc_balanced | 2.405 | 2.309 | 3.661 | 415.8 |
| tfidf_sgd_modified_huber | 2.213 | 2.142 | 2.790 | 451.9 |
| tfidf_ridge_balanced | 2.237 | 2.151 | 2.776 | 447.0 |
| tfidf_complement_nb | 2.317 | 2.255 | 2.954 | 431.6 |

- Environment: Intel(R) Xeon(R) Processor @ 2.60GHz (2 cores); Python 3.11.2; records 299 in locked-test file order; timer time.perf_counter.
- Files: `timing/end_to_end_timing.csv` (aggregate only), `timing/timing_manifest.json`, `timing/timing_protocol.md`. Raw timing files are intentionally not saved.

## 9. PII and human-validation status

- **Synthetic PII stress test** (separate from human validation; 100 unchanged locked-test records per type, all 5 seeds):
  - redaction-rule recall: **1.0000** (configured rules remove all injected synthetic PII of the four configured families).
  - student pii_risk head recall on injected views: overall **0.262** (email 0.878, phone 0.040, patient_id 0.040, ssn 0.090) — limited on out-of-distribution synthetic injections appended to clean text; honest limitation.
  - student pii_risk head `none_or_redacted` rate on redacted views: **0.980**.
  - In-distribution pii_risk task macro-F1 on locked test: 0.4978 (reported separately in section 6).
  - This stress test verifies coverage of the CONFIGURED pattern families only; it is not evidence of general named-entity/address/PII recognition.
- **Independent human PII validation: BLOCKED** — no independent human labels exist; per-type precision/recall/F1/support vs human labels and de-identified false negatives are BLOCKED; no agreement statistics were fabricated. Only `pii/human_annotation_template.csv` was created (one clearly marked TEMPLATE EXAMPLE row).
- **LLM provenance: ABSENT** — no provider, model/version, dates, generation settings, prompts, schema, retries, or adjudication logs exist for the submitted model (`pii/llm_provenance_status.json`). The dataset labels are LLM-assisted weak supervision as described in the manuscript, but none of the provenance artifacts are present.
- **Ethics records: ABSENT** — no ethics approval/exemption records are present in the package.

## 10. BLOCKED / ABSENT items (verified vs blocked — nothing fabricated)

**Verified in this run:**

1. Input verification: 2,000 records; SHA-256 match; manuscript/letter/plan readable; all reference files present; columns and label values recorded.
2. Corrected symmetric-KL gradient check: PASS (finite differences).
3. Leakage audit + group-aware locked split: 0 group / exact-pair / near-pair crossings.
4. Validation-only lambda selection under the preregistered rule; locked-test evaluation exactly once per seed (5 seeds), with bootstrap CIs, per-class metrics, subgroup diagnostics, rare-class support, changed/unchanged consistency.
5. TF-IDF baselines with validation-only parameter selection, same locked split.
6. Common end-to-end timing for the student and all runnable baselines.
7. Configured synthetic PII stress test (separate from human validation).

**BLOCKED / ABSENT (not fabricated):**

1. **Exact submitted implementation**: NOT_FOUND in the workspace -> this run is labeled **CORRECTED_FAITHFUL_REIMPLEMENTATION**, not an exact reproduction.
2. **Verification of the submitted reference metrics** (composite macro-F1 0.6896, redaction consistency 0.9867, latency 0.2042 ms/record): no original code, weights, predictions, or logs exist to verify them; they appear only as manuscript text. This run's locked-test results (composite 0.5421, consistency 0.9679) come from the corrected reimplementation on a NEW group-aware split and are not comparable 1:1 with the submitted numbers.
3. **Compact multilingual encoder baseline**: BLOCKED (torch/sentence-transformers not installed; CPU-only environment). No substitute architecture.
4. **Frozen sentence-embedding baselines** (MiniLM-L12, multilingual-E5-small, BGE-small-zh): not runnable (sentence-transformers not installed); submitted numbers not verifiable.
5. **Independent human PII validation**: BLOCKED (no human labels).
6. **LLM provenance metadata** (provider, model/version, dates, settings, prompts, schema, retries, adjudication logs): ABSENT.
7. **Human agreement statistics** (human-LLM agreement, inter-rater agreement): BLOCKED (no human labels).
8. **Ethics approval/exemption records**: ABSENT.
9. **Hospital/website/account/collection-batch/synthetic-parent metadata**: ABSENT (columns not present in the dataset); source-level grouping beyond duplicate lineage is not possible.

## 11. Manuscript update map (experiment artifacts only; the manuscript is NOT modified in this run)

| Manuscript section | Update required |
|---|---|
| Section 3.3 (method/implementation) | State that the released code is a corrected faithful reimplementation (pure numpy/scipy/sklearn) of the TF-IDF student; the exact submitted implementation was unavailable. Document the recorded assumptions (`implementation_assumptions.md`), including the corrected symmetric-KL gradient and the 160 full-batch-epoch schedule. |
| Section 3.3/4.1 hyperparameters | The manuscript reports 160 full-batch epochs, Adam lr=0.006, wd=1e-4, seed 20260715; this run confirms that schedule and adds seeds 20260715-20260719. The reference code's 40-epoch/mini-batch schedule must not be described as the manuscript schedule. |
| Section 4.1 data/split | The submitted 1,400/300/300 split is replaced by a group-aware locked split (train/validation/test = 1400/301/299, seed 20260720) after a duplicate/near-duplicate audit (120 exact pairs, 448 near pairs, 0 crossings). Describe the leakage audit; note that hospital/website/account/synthetic parent metadata are absent. |
| Section 4.3 main results | Replace/annotate with this run's locked-test results (composite 0.5421 ± 0.0077 across 5 seeds; consistency all/changed/unchanged 0.9679/0.9381/1.0000). The submitted reference metrics (0.6896 / 0.9867) could not be verified and must not be presented as reproduced. |
| Section 4.3/4.5 lambda selection | Report the preregistered validation-only rule and the selected lambda (0.25); remove the post-hoc lambda=0.10 test-set comparison from the confirmatory analysis. |
| Section 4.2/4.3 baselines | Report validation-selected hyperparameters (`baselines/hyperparameters.csv`) and locked-test results (`baselines/locked_test_results.csv`). The compact multilingual encoder and frozen embedding baselines are BLOCKED/not runnable in this environment — list as limitation, not as results. |
| Section 4.3 subgroup analysis | Add the joint language/source diagnostic (en 0.5747 vs zh 0.4001); language and source_family are perfectly confounded — do not claim separate causal effects. |
| Section 4.3 rare classes | medication support is only 4 in the locked test (F1 0.3333); report per-class support with any rare-class claim (`model/per_class_metrics.csv`). |
| Section 4.4 consistency | Report all/changed/unchanged consistency separately (0.9679 / 0.9381 / 1.0000); unchanged-record consistency is trivially 1.0 and must not be presented as evidence alone. |
| Section 4.4/4.5 PII | Separate (a) in-distribution pii_risk task metrics (0.4978), (b) the configured synthetic PII stress test (rule recall 1.00 on configured families; pii-head recall on out-of-distribution injections limited), and (c) independent human PII validation — BLOCKED. |
| Section 4.5 latency | End-to-end CPU latency here is 1.526 ms/record mean (full pipeline: normalization -> redaction -> TF-IDF -> inference -> decode -> serialize; threads=1; see `timing/timing_protocol.md`). The submitted 0.2042 ms/record model-level figure could not be verified (different hardware/pipeline definition); report hardware and pipeline boundary with any latency claim. |
| Section 3.2/4.1 provenance | LLM provenance (provider, model/version, settings, prompts, logs) is ABSENT; state the actual labeling provenance (LLM-assisted weak supervision per the manuscript) and the human-validation status (BLOCKED) explicitly. |
| Ethics statement | Ethics approval/exemption records are ABSENT; the manuscript must not claim an exemption that is not documented. |
| Reproducibility | Ship `reproducibility/` (requirements.txt, run_commands.txt, manifest.csv, SHA256SUMS.txt, summary.json) and note that exact reproduction requires the original submitted code, which was unavailable. |

## 12. Output tree

```
major_revision_run_v2/
├── FINAL_REVISION_RUN_REPORT.md
├── implementation_assumptions.md
├── src/                  corrected runnable code (common, metrics_utils, student,
│                         gradient_check, stage1..stage7)
├── data/                 input_verification.{json,md}, hashes.json, leakage_audit.{csv,json,md},
│                         locked_split.csv, locked_split_manifest.json
├── model/                selection_config.yaml, selection_log.{json,md},
│                         validation_lambda_results.csv, locked_test_metrics_by_seed.csv,
│                         locked_test_summary_across_seeds.csv, locked_test_predictions.csv,
│                         per_class_metrics.csv, bootstrap_confidence_intervals.csv
├── baselines/            hyperparameters.csv, validation_results.csv, locked_test_results.csv,
│                         locked_test_summary_across_seeds.csv, per_class_results.csv,
│                         bootstrap_confidence_intervals.csv, encoder_baseline_status.json
├── timing/               end_to_end_timing.csv, timing_manifest.json, timing_protocol.md
├── pii/                  synthetic_pii_stress_test_summary.csv, human_annotation_template.csv,
│                         pii_and_human_validation.md, llm_provenance_status.json
├── gradient_check/       symmetric_kl_logit_gradients.csv, parameter_gradient_check.csv,
│                         gradient_check_results.json, GRADIENT_CHECK.md
├── logs/                 stage run logs (stage2_training/*.log)
└── reproducibility/      README.md, requirements.txt, run_commands.txt, manifest.csv,
                          summary.json, SHA256SUMS.txt, final_output_summary.txt
```

Storage constraints honoured: no checkpoints, no `tfidf_vectorizer.pkl`, no `fitted_models.pkl`, no per-baseline prediction files, no raw timing files, no figures/notebooks/caches/wheels, no duplicate datasets. Models are fitted and released one at a time, in memory.

