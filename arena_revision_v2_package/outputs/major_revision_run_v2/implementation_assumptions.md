# Implementation Assumptions — CORRECTED_FAITHFUL_REIMPLEMENTATION

Run label: **CORRECTED_FAITHFUL_REIMPLEMENTATION** (manuscript IJSEKE-D-26-00185).
The exact submitted implementation is **not available** in this workspace, so this
run re-implements the manuscript Section 3.3 specification from scratch in pure
numpy/scipy/scikit-learn. It is a **corrected faithful reimplementation, not an
exact reproduction**. Every detail that the manuscript leaves unspecified is
recorded here.

## Corrections applied relative to `code_reference/` (the shipped reference)

1. **Symmetric-KL backward pass (REQUIRED CORRECTION).** The reference
   `code_reference/student.py` uses the one-directional approximation
   `dlogits_o += lam*0.5*(p_o - p_r)`, `dlogits_r += lam*0.5*(p_r - p_o)`, which is
   **not** the gradient of `0.5*(KL(p||q) + KL(q||p))`. This run implements the
   complete batch-mean logit gradients with epsilon clipping:
   `g_p = 0.5/B * [ p*(log(p)-log(q)-KL(p||q)) + (p-q) ]`,
   `g_q = 0.5/B * [ q*(log(q)-log(p)-KL(q||p)) + (q-p) ]`
   (see `src/student.py::symmetric_kl_logit_grads`). Verified by a finite-difference
   gradient check (`gradient_check/`, PASS required before training).
2. **Training schedule (REQUIRED CORRECTION).** The reference `common.py` ships
   `EPOCHS = 40` with `BATCH_SIZE = 256`. The manuscript states **160 full-batch
   epochs**. This run uses **160 full-batch epochs** (one Adam update per epoch over
   the whole locked train split; no mini-batching, no silent substitution).
3. **Run root.** The reference points at `/home/user/revision_runs/major_revision_run`
   and reads uploads from `/home/user/uploads`. This run reads the preserved inputs
   from `arena_revision_v2_package/input/` (SHA-256 verified) and writes only under
   `arena_revision_v2_package/outputs/major_revision_run_v2/`.

## Manuscript-specified settings (implemented exactly)

- Paired original/redacted text views; deterministic local redaction
  (`+1-555-XXXX` phone, email, `PID-XXXXXX`, `XXX-XX-XXXX` SSN patterns — identical
  to `code_reference/redaction.py`).
- char_wb TF-IDF ngrams (2,5), max 6,000 features; word TF-IDF ngrams (1,2),
  max 4,000 features; concatenated input (10,000-dimensional when the fitted
  vocabularies allow; the actual dimension is recorded in the outputs).
- TF-IDF fitted on original + redacted TRAIN views only.
- 384-dimensional shared layer; dropout 0.15 (inverted, training only).
- Four heads: service_domain (10), sentiment (3), priority (3), pii_risk (2).
- Supervised multi-task cross-entropy: all four heads on the original view; the
  redacted view shares the three semantic targets and uses its own pii_risk target
  (`pii_redacted`). Task weights alpha_t = 1 for all tasks (manuscript).
- Symmetric KL consistency `0.5*(KL(p||q)+KL(q||p))` for service_domain, sentiment,
  priority ONLY (pii_risk excluded, per manuscript).
- Total loss = sum of supervised CE (both views) + lambda * sum of symmetric KL.
- 160 full-batch epochs; Adam lr 0.006; weight decay 1e-4; seeds 20260715-20260719.
- Lambda grid {0, 0.03, 0.05, 0.10, 0.25}; validation-only selection rule
  (see `model/selection_config.yaml`).

## Genuinely unspecified details (assumptions, recorded)

1. **Shared-layer activation**: ReLU (manuscript does not name it).
2. **Weight decay style**: decoupled (AdamW-style), as in the reference code.
3. **Adam betas / epsilon**: (0.9, 0.999), eps 1e-8 (defaults; manuscript silent).
4. **Initialization**: He for the shared projection (ReLU), Glorot-style for heads;
   `numpy.default_rng(seed)` (reference convention).
5. **TF-IDF `min_df=2`** for both vectorizers (reference convention; manuscript
   specifies only ngram ranges and max features).
6. **Dropout mask RNG**: `numpy.default_rng(seed + 1)` (reference convention).
7. **Split protocol** (manuscript does not prescribe one): group-aware locked split,
   seed 20260720, targets 70/15/15, stratified by language, greedy largest-first
   assignment; exact duplicates = NFKC + casefold + whitespace collapse on both
   views; near duplicates = char_wb (3,5) TF-IDF cosine >= 0.90 on both views.
8. **Baseline training view**: TF-IDF baselines are trained on the redacted train
   view and evaluated on both views (submitted-script convention; metrics reported
   on the redacted view, the inference-time convention). The logistic-regression
   baseline uses `OneVsRestClassifier(LogisticRegression(solver="liblinear"))`
   because scikit-learn >= 1.5 no longer implicitly wraps multiclass liblinear in
   one-vs-rest; the explicit wrapper reproduces the historical OvR behaviour.
9. **Bootstrap**: 2,000 paired replicates, percentile 95% intervals, seed 20260721.
10. **Timing protocol**: one common end-to-end boundary per record (normalization,
    local redaction, TF-IDF construction, inference, decoding, JSON serialization),
    threads=1, 1 warm-up pass + 5 measured repetitions over the locked test records.
11. **Storage constraints**: no checkpoints / vectorizer pickle / fitted-model
    pickle / per-baseline prediction files / raw timing files are persisted; models
    are refit deterministically in memory in each stage that needs them.
12. **Determinism**: TfidfVectorizer fitting is deterministic; student init,
    dropout masks and bootstrap resampling are seeded; results are reproducible
    given the same numpy/scipy/scikit-learn versions (see
    `reproducibility/requirements.txt`).
