# Reproducibility Package — RC-EdgeDistill Major Revision Run v2

Run label: **CORRECTED_FAITHFUL_REIMPLEMENTATION** — a **corrected faithful reimplementation** (the exact submitted
implementation is unavailable; the reference symmetric-KL backward pass was corrected and
the manuscript's 160 full-batch-epoch schedule was used). See
`../implementation_assumptions.md` and `../FINAL_REVISION_RUN_REPORT.md`.

## Contents
- `requirements.txt` — pinned runtime versions (numpy/pandas/scipy/scikit-learn; no torch needed).
- `run_commands.txt` — exact commands for every stage, in order.
- `manifest.csv` — SHA-256 and size for every run output file.
- `SHA256SUMS.txt` — `sha256sum`-format checksums of the whole run directory.
- `summary.json` — machine-readable key results, hashes, gradient-check result, and BLOCKED items.
- `final_output_summary.txt` — output tree, file count, total bytes.

## Data / code hashes
- Data SHA-256: `bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e` (verified against the expected value in stage 1)
- Code SHA-256 (src/*.py): `fdc9a792c713446b27ffe746d5687ad39d0967425865ce377cb2b0d3a5817110`
- Split seed: 20260720; model seeds: [20260715, 20260716, 20260717, 20260718, 20260719]

## Gradient check
- Symmetric-KL logit gradients: max abs error 3.036e-10 -> PASS
- Student parameter gradients: max abs error 2.949e-09 -> PASS
- Overall: **PASS**

## Selected lambda
- 0.25 (preregistered rule in `../model/selection_config.yaml`; validation only)

## Key locked-test result (5 seeds, redacted view)
- composite macro-F1: 0.5421 ± 0.0077
- mean consistency (all / changed / unchanged): 0.9679 / 0.9381 / 1.0000

## Blocked items (not fabricated)
1. Exact submitted implementation (NOT_FOUND -> CORRECTED_FAITHFUL_REIMPLEMENTATION)
2. Verification of submitted reference metrics 0.6896 / 0.9867 / 0.2042 (no original artifacts)
3. Compact multilingual encoder baseline (torch/sentence-transformers not installed)
4. Frozen sentence-embedding baselines (sentence-transformers not installed; not verifiable)
5. Independent human PII validation (no human labels)
6. LLM provenance metadata (ABSENT)
7. Human agreement statistics (BLOCKED)
8. Ethics records (ABSENT)
9. Hospital/website/account/synthetic-parent metadata (ABSENT)

See `../FINAL_REVISION_RUN_REPORT.md` for the full report.
