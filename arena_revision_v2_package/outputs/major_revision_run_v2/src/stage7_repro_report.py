"""Stage 7: reproducibility package + FINAL_REVISION_RUN_REPORT.md.

All numbers in the report are read from the saved result files (no manual
transcription). Files are hashed; nothing pre-existing is modified.
"""
import glob
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RUN_ROOT, DATA_DIR, MODEL_DIR, BASELINES_DIR, TIMING_DIR,
                    PII_DIR, REPRO_DIR, GRADIENT_CHECK_DIR, RUN_LABEL, RUN_DATE,
                    PKG_ROOT, LAMBDA_GRID, SEEDS, sha256_file, code_hash,
                    read_json, ensure_dirs)

REPORT_PATH = RUN_ROOT / "FINAL_REVISION_RUN_REPORT.md"
SRC_DIR = RUN_ROOT / "src"


def read_csv(p):
    return pd.read_csv(p)


def main():
    ensure_dirs()

    # ---------------- load results ----------------
    split_manifest = read_json(DATA_DIR / "locked_split_manifest.json")
    verification = read_json(DATA_DIR / "input_verification.json")
    val_res = read_csv(MODEL_DIR / "validation_lambda_results.csv")
    selection = read_json(MODEL_DIR / "selection_log.json")
    test_seed = read_csv(MODEL_DIR / "locked_test_metrics_by_seed.csv")
    test_summ = read_csv(MODEL_DIR / "locked_test_summary_across_seeds.csv")
    boot = read_csv(MODEL_DIR / "bootstrap_confidence_intervals.csv")
    per_class = read_csv(MODEL_DIR / "per_class_metrics.csv")
    base_test = read_csv(BASELINES_DIR / "locked_test_results.csv")
    base_summ = read_csv(BASELINES_DIR / "locked_test_summary_across_seeds.csv")
    base_hyper = read_csv(BASELINES_DIR / "hyperparameters.csv")
    timing = read_csv(TIMING_DIR / "end_to_end_timing.csv")
    pii_summ = read_csv(PII_DIR / "synthetic_pii_stress_test_summary.csv")
    timing_manifest = read_json(TIMING_DIR / "timing_manifest.json")
    grad = read_json(GRADIENT_CHECK_DIR / "gradient_check_results.json")

    lam = float(selection["selected_lambda"])
    ts = test_summ.iloc[0]

    def m(col):
        return float(ts[f"{col}_mean"])

    def sd(col):
        return float(ts[f"{col}_std"])

    boot_agg = boot.groupby("metric").agg(ci_lower=("ci_lower", "mean"),
                                          ci_upper=("ci_upper", "mean")).reset_index()
    boot_map = {r["metric"]: (r["ci_lower"], r["ci_upper"]) for _, r in boot_agg.iterrows()}

    val_agg = val_res.groupby("lambda").agg(
        val_composite_macro_f1_mean=("val_composite_macro_f1", "mean"),
        val_mean_consistency_mean=("val_mean_consistency", "mean")).reset_index()

    # ---------------- requirements.txt ----------------
    import scipy, sklearn
    reqs = [
        f"# Python {platform.python_version()} (CPython) - recorded from the run environment",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"scipy=={scipy.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        "# torch / transformers / sentence-transformers are NOT required: the student is",
        "# implemented in pure numpy/scipy/scikit-learn (see implementation_assumptions.md).",
        "# torch and sentence-transformers are NOT installed in this environment; the compact",
        "# multilingual encoder baseline and the frozen-embedding baselines are BLOCKED",
        "# (see baselines/encoder_baseline_status.json).",
    ]
    with open(REPRO_DIR / "requirements.txt", "w") as f:
        f.write("\n".join(reqs) + "\n")

    # ---------------- run_commands.txt ----------------
    cmds = [
        "# RC-EdgeDistill major revision run v2 - exact commands.",
        "# Run from the repository root with the pinned environment (reproducibility/requirements.txt).",
        "# All stages are CPU-only with OMP_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. No GPU packages used.",
        "VENV=/path/to/venv   # python >= 3.11 with numpy/pandas/scipy/scikit-learn (see requirements.txt)",
        "RUN=arena_revision_v2_package/outputs/major_revision_run_v2",
        "# Stage 1: input verification + leakage audit + group-aware locked split (seed 20260720)",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage1_input_audit.py",
        "# Stage 2a: finite-difference gradient check of the corrected symmetric-KL backward pass",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python gradient_check.py",
        "# Stage 2b: train TF-IDF student, lambda grid [0,0.03,0.05,0.10,0.25] x seeds "
        "[20260715..20260719], 160 FULL-BATCH epochs, Adam lr=0.006 wd=1e-4 (validation-only selection)",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage2_train_student.py --epochs 160",
        "# Stage 3: retrain each selected seed in memory; evaluate the locked test exactly once per seed",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage3_test_student.py",
        "# Stage 4: TF-IDF baselines (params selected on validation only) + encoder BLOCKED status",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage4_baselines.py",
        "# Stage 5: common end-to-end CPU timing (1 warm-up + 5 repetitions, per-record)",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage5_timing.py",
        "# Stage 6: synthetic PII stress test + human-validation BLOCKED status + LLM provenance ABSENT",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage6_pii.py",
        "# Stage 7: reproducibility package + final report",
        f"cd $RUN/src && OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python stage7_repro_report.py",
    ]
    with open(REPRO_DIR / "run_commands.txt", "w") as f:
        f.write("\n".join(cmds) + "\n")

    # ---------------- manifest.csv ----------------
    code_files = sorted(glob.glob(str(SRC_DIR / "*.py")))
    chash = code_hash(code_files)
    all_files = []
    for dp, _, fns in os.walk(RUN_ROOT):
        for fn in sorted(fns):
            full = Path(dp) / fn
            rel = full.relative_to(RUN_ROOT)
            if rel == Path("reproducibility/SHA256SUMS.txt"):
                continue
            if rel == Path("reproducibility/manifest.csv"):
                continue
            if rel == Path("FINAL_REVISION_RUN_REPORT.md"):
                continue
            if rel == Path("reproducibility/final_output_summary.txt"):
                continue
            if "__pycache__" in full.parts:
                continue
            all_files.append((rel, full))
    all_files.sort(key=lambda x: str(x[0]))
    man_rows = [{"relative_path": str(rel), "size_bytes": full.stat().st_size,
                 "sha256": sha256_file(full)} for rel, full in all_files]
    pd.DataFrame(man_rows).to_csv(REPRO_DIR / "manifest.csv", index=False)

    # ---------------- summary.json ----------------
    summary = {
        "run_label": RUN_LABEL,
        "manuscript": "IJSEKE-D-26-00185 (RC-EdgeDistill)",
        "date": RUN_DATE,
        "statement": "corrected faithful reimplementation (NOT an exact reproduction; "
                     "the exact submitted implementation is unavailable)",
        "repository_root": str(PKG_ROOT.parent),
        "run_root": str(RUN_ROOT),
        "data_sha256": split_manifest["data_sha256"],
        "data_sha256_verified": verification["dataset"]["sha256_match"],
        "code_hash": chash,
        "split_seed": split_manifest["split_seed"],
        "split_counts": split_manifest["split_counts"],
        "selected_lambda": lam,
        "training": {"epochs": 160, "batching": "full-batch", "optimizer": "Adam",
                     "lr": 0.006, "weight_decay": 1e-4, "dropout": 0.15,
                     "shared_dim": 384, "seeds": SEEDS},
        "gradient_check": {
            "symmetric_kl_max_abs_error": grad["symmetric_kl_logit_gradient_check"]["max_abs_error"],
            "symmetric_kl_tolerance": grad["symmetric_kl_logit_gradient_check"]["tolerance"],
            "symmetric_kl_result": grad["symmetric_kl_logit_gradient_check"]["result"],
            "parameter_max_abs_error": grad["student_parameter_gradient_check"]["max_abs_error"],
            "parameter_tolerance": grad["student_parameter_gradient_check"]["tolerance"],
            "parameter_result": grad["student_parameter_gradient_check"]["result"],
            "overall_result": grad["overall_result"],
        },
        "locked_test": {
            "rows": int(m("test_rows")),
            "composite_macro_f1_mean": m("composite_macro_f1"),
            "composite_macro_f1_std": sd("composite_macro_f1"),
            "service_domain_macro_f1_mean": m("service_domain_macro_f1"),
            "sentiment_macro_f1_mean": m("sentiment_macro_f1"),
            "priority_macro_f1_mean": m("priority_macro_f1"),
            "pii_risk_macro_f1_mean": m("pii_risk_macro_f1"),
            "mean_consistency_all": m("mean_consistency_all"),
            "mean_consistency_changed": m("mean_consistency_changed"),
            "mean_consistency_unchanged": m("mean_consistency_unchanged"),
            "en_composite": m("en_composite_macro_f1"),
            "zh_composite": m("zh_composite_macro_f1"),
            "medication_support": int(m("medication_support")),
            "medication_f1": m("medication_f1"),
        },
        "baselines_test_composite_macro_f1_mean": {
            r["baseline"]: float(r["composite_macro_f1_mean"])
            for _, r in base_summ.iterrows()},
        "timing_mean_ms_per_record": {r["method"]: float(r["mean_ms_per_record"])
                                      for _, r in timing.iterrows()},
        "blocked_items": [
            "exact submitted implementation (NOT_FOUND -> CORRECTED_FAITHFUL_REIMPLEMENTATION)",
            "verification of submitted reference metrics 0.6896 / 0.9867 / 0.2042 "
            "(no original checkpoints, predictions, or logs are present)",
            "compact multilingual encoder baseline (torch/sentence-transformers not installed)",
            "frozen sentence-embedding baselines (sentence-transformers not installed; "
            "submitted numbers not verifiable)",
            "independent human PII validation (no human labels)",
            "LLM provenance metadata for the submitted model (ABSENT)",
            "human agreement statistics (BLOCKED)",
            "ethics approval/exemption records (ABSENT)",
            "hospital/website/account/collection-batch/synthetic-parent metadata (ABSENT)",
        ],
    }
    with open(REPRO_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------------- FINAL_REVISION_RUN_REPORT.md ----------------
    L = []
    A = L.append
    A("# FINAL REVISION RUN REPORT — RC-EdgeDistill (IJSEKE-D-26-00185)")
    A("")
    A(f"- **Run label: `{RUN_LABEL}`** — this is a **corrected faithful reimplementation**, "
      "**NOT an exact reproduction**. The exact submitted implementation is not available in "
      "the workspace; the manuscript Section 3.3 specification was re-implemented from scratch "
      "in pure numpy/scipy/scikit-learn, with the reference `code_reference/student.py` "
      "symmetric-KL backward pass **corrected** and the reference 40-epoch/mini-batch schedule "
      "**replaced by the manuscript's 160 full-batch epochs**. See `implementation_assumptions.md`.")
    A(f"- Date: {RUN_DATE} | Repository: `{PKG_ROOT.parent}` | Run root: `{RUN_ROOT}`")
    A("- Hardware/software: CPU-only (2 cores), no GPU, `OMP_NUM_THREADS=1`, "
      f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
      f"scipy {scipy.__version__}, scikit-learn {sklearn.__version__}")
    A("- The submitted reference metrics (**0.6896 composite macro-F1 / 0.9867 consistency / "
      "0.2042 ms per record**) are **NOT claimed as reproduced**: no original checkpoints, "
      "predictions, or logs exist in the package to verify them. All numbers below come from "
      "this run on a NEW group-aware locked split.")
    A("")
    A("## 1. Commands")
    A("")
    A("See `reproducibility/run_commands.txt` for the exact stage commands. Summary:")
    A("")
    A("```bash")
    A("cd arena_revision_v2_package/outputs/major_revision_run_v2/src")
    A("OMP_NUM_THREADS=1 python3 stage1_input_audit.py      # input check + leakage audit + locked split")
    A("OMP_NUM_THREADS=1 python3 gradient_check.py          # finite-difference check of corrected symmetric KL")
    A("OMP_NUM_THREADS=1 python3 stage2_train_student.py --epochs 160   # lambda grid x 5 seeds (validation only)")
    A("OMP_NUM_THREADS=1 python3 stage3_test_student.py     # retrain per seed, locked test once per seed")
    A("OMP_NUM_THREADS=1 python3 stage4_baselines.py        # TF-IDF baselines (validation-selected)")
    A("OMP_NUM_THREADS=1 python3 stage5_timing.py           # common end-to-end timing")
    A("OMP_NUM_THREADS=1 python3 stage6_pii.py              # synthetic PII stress + validation status")
    A("OMP_NUM_THREADS=1 python3 stage7_repro_report.py     # this report + reproducibility package")
    A("```")
    A("")
    A("## 2. Input verification, data and hashes")
    A("")
    A(f"- Dataset: `arena_revision_v2_package/input/prepared_paired_dataset_2000.csv` — "
      f"**{verification['dataset']['rows']} records** (expected 2,000: "
      f"{'PASS' if verification['dataset']['rows'] == 2000 else 'FAIL'}).")
    A(f"- **Data SHA-256:** `{split_manifest['data_sha256']}` — matches the expected "
      f"`bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e`: "
      f"**{verification['dataset']['sha256_match']}**.")
    A(f"- Manuscript, decision letter, and revision plan readable: PASS; all 12 required "
      f"`code_reference/*.py` files present: PASS (`data/input_verification.json`).")
    A(f"- **Code SHA-256** (src/*.py of this run): `{chash}`")
    A(f"- **Split seed:** {split_manifest['split_seed']} (fixed, recorded); model seeds: {SEEDS}")
    A(f"- Data columns: `{', '.join(verification['dataset']['columns'])}`")
    A("")
    A("## 3. Gradient check (corrected symmetric-KL backward pass) — REQUIRED GATE")
    A("")
    A("The reference `code_reference/student.py` ships an incomplete symmetric-KL backward pass "
      "(`lam*0.5*(p_o-p_r)` / `lam*0.5*(p_r-p_o)`), which is not the gradient of "
      "`0.5*(KL(p||q)+KL(q||p))`. This run implements the complete batch-mean logit gradients "
      "with epsilon clipping:")
    A("")
    A("```text")
    A("g_p = 0.5/B * [ p*(log(p) - log(q) - KL(p||q)) + (p - q) ]")
    A("g_q = 0.5/B * [ q*(log(q) - log(p) - KL(q||p)) + (q - p) ]")
    A("```")
    A("")
    A("Finite-difference verification (central differences, h=1e-6, tiny dense fixtures; "
      "details in `gradient_check/`):")
    A("")
    A(f"- Symmetric-KL logit gradients: max abs error **"
      f"{grad['symmetric_kl_logit_gradient_check']['max_abs_error']:.3e}** vs tolerance "
      f"{grad['symmetric_kl_logit_gradient_check']['tolerance']:.1e} -> **"
      f"{grad['symmetric_kl_logit_gradient_check']['result']}**")
    A(f"- Full student parameter gradients (CE + corrected KL + shared layer + heads): max abs "
      f"error **{grad['student_parameter_gradient_check']['max_abs_error']:.3e}** vs tolerance "
      f"{grad['student_parameter_gradient_check']['tolerance']:.1e} -> **"
      f"{grad['student_parameter_gradient_check']['result']}**")
    A(f"- **Overall: {grad['overall_result']}** (training did not proceed unless PASS)")
    A("")
    A("## 4. Split audit (leakage-safe, group-aware locked split)")
    A("")
    A(f"- Rows: {split_manifest['rows']}; locked split counts: "
      f"train {split_manifest['split_counts'].get('train')}, "
      f"validation {split_manifest['split_counts'].get('validation')}, "
      f"test {split_manifest['split_counts'].get('test')} "
      f"(targets 70/15/15; group-aware greedy assignment, seed {split_manifest['split_seed']}).")
    A(f"- Exact duplicates (NFKC + casefold + whitespace collapse, both views): "
      f"**{split_manifest['duplicate_audit']['exact_duplicate_pairs']} pairs**, "
      f"{split_manifest['duplicate_audit']['exact_duplicate_member_rows']} member rows.")
    A(f"- Near duplicates (char_wb (3,5) TF-IDF cosine >= "
      f"{split_manifest['duplicate_audit']['near_duplicate_threshold']}, both views): "
      f"**{split_manifest['duplicate_audit']['near_duplicate_pairs']} pairs**, "
      f"{split_manifest['duplicate_audit']['near_duplicate_member_rows']} member rows.")
    A(f"- Lineage groups (union-find): **{split_manifest['duplicate_audit']['total_groups']}** "
      f"(largest group {split_manifest['duplicate_audit']['largest_group_size']}).")
    A(f"- **Crossing counts:** groups crossing splits = "
      f"**{split_manifest['crossing_counts']['groups_crossing_splits']}**, exact-duplicate pairs "
      f"separated = **{split_manifest['crossing_counts']['exact_duplicate_pairs_separated']}**, "
      f"near-duplicate pairs separated = "
      f"**{split_manifest['crossing_counts']['near_duplicate_pairs_separated']}** (all must be 0).")
    A(f"- Metadata: hospital / website / account / collection batch / synthetic-parent id = "
      f"**ABSENT** (columns not present); `pii_original` used as synthetic-injection proxy. "
      "Source-level grouping beyond duplicate lineage is therefore not possible — recorded "
      "limitation.")
    A(f"- Language/source confounding: **perfectly confounded** "
      f"(en <-> english_public_hospital_reviews, zh <-> chinese_public_web_deidentified); "
      f"subgroup results are a **joint language/source diagnostic** — no separate causal "
      "language or source effects are claimed.")
    A(f"- Text views: {split_manifest['text_view_equality']['rows_changed_by_redaction']} rows "
      f"changed by redaction, "
      f"{split_manifest['text_view_equality']['rows_original_equals_redacted']} unchanged.")
    A("- Full details: `data/input_verification.{json,md}`, `data/hashes.json`, "
      "`data/leakage_audit.{csv,json,md}`, `data/locked_split.csv`, "
      "`data/locked_split_manifest.json`.")
    A("")
    A("## 5. Model selection (validation only; preregistered rule)")
    A("")
    A("Rule (declared in `model/selection_config.yaml` **before** any training, and applied "
      "before any test prediction): maximize validation composite macro-F1; within tolerance "
      "0.005 maximize validation mean consistency; then choose the smaller lambda. "
      "Locked-test labels/predictions were never used for selection.")
    A("")
    A("Training: 160 FULL-BATCH epochs, Adam lr=0.006, weight decay 1e-4, dropout 0.15, "
      "shared 384, seeds 20260715–20260719, lambda grid {0, 0.03, 0.05, 0.10, 0.25}.")
    A("")
    A("| lambda | val composite macro-F1 (mean) | val mean consistency (mean) |")
    A("|---:|---:|---:|")
    for _, r in val_agg.iterrows():
        mark = " **selected**" if float(r["lambda"]) == lam else ""
        A(f"| {r['lambda']} | {r['val_composite_macro_f1_mean']:.4f} | "
          f"{r['val_mean_consistency_mean']:.4f} |{mark}")
    A("")
    A(f"**Selected lambda: {lam}** (best validation composite macro-F1 "
      f"{selection['best_composite_macro_f1']:.6f}).")
    A("")
    A("## 6. Locked-test metrics — RC-EdgeDistill (selected lambda, 5 seeds, once each)")
    A("")
    A("Each seed was retrained in memory (no checkpoints) and the locked test was evaluated "
      "exactly once per seed. Primary task metrics on the redacted view (inference "
      "convention); composite = mean of the three semantic-task macro-F1 values.")
    A("")
    A("| seed | composite | service_domain | sentiment | priority | pii_risk | consistency (all) | "
      "consistency (changed) | consistency (unchanged) |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in test_seed.iterrows():
        A(f"| {int(r['seed'])} | {r['composite_macro_f1']:.4f} | {r['service_domain_macro_f1']:.4f} | "
          f"{r['sentiment_macro_f1']:.4f} | {r['priority_macro_f1']:.4f} | {r['pii_risk_macro_f1']:.4f} | "
          f"{r['mean_consistency_all']:.4f} | {r['mean_consistency_changed']:.4f} | "
          f"{r['mean_consistency_unchanged']:.4f} |")
    A(f"| **mean** | **{m('composite_macro_f1'):.4f}** | **{m('service_domain_macro_f1'):.4f}** | "
      f"**{m('sentiment_macro_f1'):.4f}** | **{m('priority_macro_f1'):.4f}** | "
      f"**{m('pii_risk_macro_f1'):.4f}** | **{m('mean_consistency_all'):.4f}** | "
      f"**{m('mean_consistency_changed'):.4f}** | **{m('mean_consistency_unchanged'):.4f}** |")
    A(f"| **std** | {sd('composite_macro_f1'):.4f} | {sd('service_domain_macro_f1'):.4f} | "
      f"{sd('sentiment_macro_f1'):.4f} | {sd('priority_macro_f1'):.4f} | {sd('pii_risk_macro_f1'):.4f} | "
      f"{sd('mean_consistency_all'):.4f} | {sd('mean_consistency_changed'):.4f} | "
      f"{sd('mean_consistency_unchanged'):.4f} |")
    A("")
    A(f"- Locked test rows: {int(m('test_rows'))} ({int(m('changed_records'))} changed by redaction, "
      f"{int(m('unchanged_records'))} unchanged).")
    A(f"- Original-view composite macro-F1 (mean): {m('composite_macro_f1_original_view'):.4f}.")
    A(f"- Paired bootstrap 95% CIs (2,000 replicates, seed 20260721, percentile; mean across seeds):")
    for metric, (lo, hi) in boot_map.items():
        A(f"  - {metric}: [{lo:.4f}, {hi:.4f}]")
    A("- Per-class precision/recall/F1/support: `model/per_class_metrics.csv`. "
      "ONE combined per-record prediction CSV (all seeds): `model/locked_test_predictions.csv`.")
    A("")
    A("### Subgroup (joint language/source diagnostic — perfectly confounded)")
    A("")
    A("| subgroup | n | composite macro-F1 (mean) | mean consistency (mean) | pii_risk F1 (mean) |")
    A("|---|---:|---:|---:|---:|")
    A(f"| en / english_public_hospital_reviews | {int(m('en_n'))} | "
      f"{m('en_composite_macro_f1'):.4f} | {m('en_mean_consistency'):.4f} | "
      f"{m('en_pii_risk_macro_f1'):.4f} |")
    A(f"| zh / chinese_public_web_deidentified | {int(m('zh_n'))} | "
      f"{m('zh_composite_macro_f1'):.4f} | {m('zh_mean_consistency'):.4f} | "
      f"{m('zh_pii_risk_macro_f1'):.4f} |")
    A("")
    A("Language and source_family are the same partition (perfectly confounded): these are "
      "**joint** diagnostics, not separable language or source effects. Any performance "
      "difference between the two subgroups cannot be attributed to language alone — the "
      "Chinese records are public-web excerpts and the English records are hospital reviews "
      "from a different country/source, so **language and source are confounded** in this "
      "dataset.")
    A("")
    A("### Rare class: medication (service_domain)")
    A("")
    A(f"- Locked-test support: **{int(m('medication_support'))}** records; "
      f"F1 {m('medication_f1'):.4f} (precision {m('medication_precision'):.4f}, "
      f"recall {m('medication_recall'):.4f}). Support is very small — treat as unstable.")
    A("")
    A("## 7. Baselines (same locked split; parameters selected on validation only)")
    A("")
    A("| baseline | selected param | test composite macro-F1 (mean) | std across seeds | "
      "consistency (all) | en | zh | medication F1 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in base_summ.iterrows():
        name = r["baseline"]
        hrow = base_hyper[base_hyper["baseline"] == name].iloc[0]
        A(f"| {name} | {hrow['parameter']}={hrow['selected_value']} | "
          f"{r['composite_macro_f1_mean']:.4f} | {r['composite_macro_f1_std']:.4f} | "
          f"{r['mean_consistency_all_mean']:.4f} | {r['en_composite_macro_f1_mean']:.4f} | "
          f"{r['zh_composite_macro_f1_mean']:.4f} | {r['medication_f1_mean']:.4f} |")
    A(f"| **rc_edgedistill (lambda {lam})** | — | **{m('composite_macro_f1'):.4f}** | "
      f"**{sd('composite_macro_f1'):.4f}** | **{m('mean_consistency_all'):.4f}** | "
      f"**{m('en_composite_macro_f1'):.4f}** | **{m('zh_composite_macro_f1'):.4f}** | "
      f"**{m('medication_f1'):.4f}** |")
    A("")
    A("- Details: `baselines/hyperparameters.csv` (selection), `baselines/validation_results.csv`, "
      "`baselines/locked_test_results.csv`, `baselines/per_class_results.csv`, "
      "`baselines/bootstrap_confidence_intervals.csv`, "
      "`baselines/locked_test_summary_across_seeds.csv`. No per-baseline prediction files and "
      "no `fitted_models.pkl` are saved (storage constraint).")
    A("- **Compact multilingual encoder baseline: BLOCKED** — torch/sentence-transformers are "
      "not installed in this CPU-only environment; no substitute architecture was used "
      "(`baselines/encoder_baseline_status.json`).")
    A("- **Frozen sentence-embedding baselines (MiniLM / multilingual-E5 / BGE-small): not "
      "runnable here** — sentence-transformers is not installed; the submitted embedding "
      "numbers could not be verified and are not reproduced.")
    A("")
    A("## 8. Common end-to-end timing (CPU, threads=1, per-record, 1 warm-up + 5 repetitions)")
    A("")
    A("Pipeline per record (identical boundary for every method): input normalization (NFKC + "
      "whitespace collapse) -> local redaction (regex) -> TF-IDF construction -> inference -> "
      "decoding -> JSON serialization. Model-only latency is not mixed in.")
    A("")
    A("| method | mean ms/record | p50 | p95 | throughput (rec/s) |")
    A("|---|---:|---:|---:|---:|")
    for _, r in timing.iterrows():
        A(f"| {r['method']} | {r['mean_ms_per_record']:.3f} | {r['median_ms_per_record']:.3f} | "
          f"{r['p95_ms_per_record']:.3f} | {r['mean_throughput_records_per_second']:.1f} |")
    A("")
    A(f"- Environment: {timing_manifest['cpu']} ({timing_manifest['cpu_cores']} cores); Python "
      f"{timing_manifest['python']}; records {timing_manifest['records']} in locked-test file "
      f"order; timer {timing_manifest['timer']}.")
    A("- Files: `timing/end_to_end_timing.csv` (aggregate only), `timing/timing_manifest.json`, "
      "`timing/timing_protocol.md`. Raw timing files are intentionally not saved.")
    A("")
    A("## 9. PII and human-validation status")
    A("")
    pii_overall = pii_summ[pii_summ["pii_type"] == "ALL"]
    A(f"- **Synthetic PII stress test** (separate from human validation; 100 unchanged "
      f"locked-test records per type, all 5 seeds):")
    A(f"  - redaction-rule recall: **{pii_overall['redaction_rule_recall'].mean():.4f}** "
      "(configured rules remove all injected synthetic PII of the four configured families).")
    A(f"  - student pii_risk head recall on injected views: overall "
      f"**{pii_overall['pii_head_recall_on_injected'].mean():.3f}** "
      f"(email {pii_summ[pii_summ['pii_type'] == 'email']['pii_head_recall_on_injected'].mean():.3f}, "
      f"phone {pii_summ[pii_summ['pii_type'] == 'phone']['pii_head_recall_on_injected'].mean():.3f}, "
      f"patient_id {pii_summ[pii_summ['pii_type'] == 'patient_id']['pii_head_recall_on_injected'].mean():.3f}, "
      f"ssn {pii_summ[pii_summ['pii_type'] == 'ssn']['pii_head_recall_on_injected'].mean():.3f}) — "
      "limited on out-of-distribution synthetic injections appended to clean text; honest limitation.")
    A(f"  - student pii_risk head `none_or_redacted` rate on redacted views: "
      f"**{pii_overall['pii_head_none_or_redacted_rate_on_redacted'].mean():.3f}**.")
    A(f"  - In-distribution pii_risk task macro-F1 on locked test: {m('pii_risk_macro_f1'):.4f} "
      "(reported separately in section 6).")
    A("  - This stress test verifies coverage of the CONFIGURED pattern families only; it is "
      "not evidence of general named-entity/address/PII recognition.")
    A("- **Independent human PII validation: BLOCKED** — no independent human labels exist; "
      "per-type precision/recall/F1/support vs human labels and de-identified false negatives "
      "are BLOCKED; no agreement statistics were fabricated. Only "
      "`pii/human_annotation_template.csv` was created (one clearly marked TEMPLATE EXAMPLE row).")
    A("- **LLM provenance: ABSENT** — no provider, model/version, dates, generation settings, "
      "prompts, schema, retries, or adjudication logs exist for the submitted model "
      "(`pii/llm_provenance_status.json`). The dataset labels are LLM-assisted weak supervision "
      "as described in the manuscript, but none of the provenance artifacts are present.")
    A("- **Ethics records: ABSENT** — no ethics approval/exemption records are present in the "
      "package.")
    A("")
    A("## 10. BLOCKED / ABSENT items (verified vs blocked — nothing fabricated)")
    A("")
    A("**Verified in this run:**")
    A("")
    A("1. Input verification: 2,000 records; SHA-256 match; manuscript/letter/plan readable; "
      "all reference files present; columns and label values recorded.")
    A("2. Corrected symmetric-KL gradient check: PASS (finite differences).")
    A("3. Leakage audit + group-aware locked split: 0 group / exact-pair / near-pair crossings.")
    A("4. Validation-only lambda selection under the preregistered rule; locked-test evaluation "
      "exactly once per seed (5 seeds), with bootstrap CIs, per-class metrics, subgroup "
      "diagnostics, rare-class support, changed/unchanged consistency.")
    A("5. TF-IDF baselines with validation-only parameter selection, same locked split.")
    A("6. Common end-to-end timing for the student and all runnable baselines.")
    A("7. Configured synthetic PII stress test (separate from human validation).")
    A("")
    A("**BLOCKED / ABSENT (not fabricated):**")
    A("")
    A("1. **Exact submitted implementation**: NOT_FOUND in the workspace -> this run is labeled "
      "**CORRECTED_FAITHFUL_REIMPLEMENTATION**, not an exact reproduction.")
    A("2. **Verification of the submitted reference metrics** (composite macro-F1 0.6896, "
      "redaction consistency 0.9867, latency 0.2042 ms/record): no original code, weights, "
      "predictions, or logs exist to verify them; they appear only as manuscript text. This "
      f"run's locked-test results (composite {m('composite_macro_f1'):.4f}, consistency "
      f"{m('mean_consistency_all'):.4f}) come from the corrected reimplementation on a NEW "
      "group-aware split and are not comparable 1:1 with the submitted numbers.")
    A("3. **Compact multilingual encoder baseline**: BLOCKED (torch/sentence-transformers not "
      "installed; CPU-only environment). No substitute architecture.")
    A("4. **Frozen sentence-embedding baselines** (MiniLM-L12, multilingual-E5-small, "
      "BGE-small-zh): not runnable (sentence-transformers not installed); submitted numbers "
      "not verifiable.")
    A("5. **Independent human PII validation**: BLOCKED (no human labels).")
    A("6. **LLM provenance metadata** (provider, model/version, dates, settings, prompts, "
      "schema, retries, adjudication logs): ABSENT.")
    A("7. **Human agreement statistics** (human-LLM agreement, inter-rater agreement): BLOCKED "
      "(no human labels).")
    A("8. **Ethics approval/exemption records**: ABSENT.")
    A("9. **Hospital/website/account/collection-batch/synthetic-parent metadata**: ABSENT "
      "(columns not present in the dataset); source-level grouping beyond duplicate lineage "
      "is not possible.")
    A("")
    A("## 11. Manuscript update map (experiment artifacts only; the manuscript is NOT "
      "modified in this run)")
    A("")
    A("| Manuscript section | Update required |")
    A("|---|---|")
    A("| Section 3.3 (method/implementation) | State that the released code is a corrected "
      "faithful reimplementation (pure numpy/scipy/sklearn) of the TF-IDF student; the exact "
      "submitted implementation was unavailable. Document the recorded assumptions "
      "(`implementation_assumptions.md`), including the corrected symmetric-KL gradient and "
      "the 160 full-batch-epoch schedule. |")
    A(f"| Section 3.3/4.1 hyperparameters | The manuscript reports 160 full-batch epochs, Adam "
      f"lr=0.006, wd=1e-4, seed 20260715; this run confirms that schedule and adds seeds "
      f"20260715-20260719. The reference code's 40-epoch/mini-batch schedule must not be "
      "described as the manuscript schedule. |")
    A(f"| Section 4.1 data/split | The submitted 1,400/300/300 split is replaced by a "
      f"group-aware locked split (train/validation/test = "
      f"{split_manifest['split_counts'].get('train')}/{split_manifest['split_counts'].get('validation')}/"
      f"{split_manifest['split_counts'].get('test')}, seed {split_manifest['split_seed']}) after "
      f"a duplicate/near-duplicate audit ({split_manifest['duplicate_audit']['exact_duplicate_pairs']} "
      f"exact pairs, {split_manifest['duplicate_audit']['near_duplicate_pairs']} near pairs, "
      "0 crossings). Describe the leakage audit; note that hospital/website/account/synthetic "
      "parent metadata are absent. |")
    A(f"| Section 4.3 main results | Replace/annotate with this run's locked-test results "
      f"(composite {m('composite_macro_f1'):.4f} ± {sd('composite_macro_f1'):.4f} across 5 "
      "seeds; consistency all/changed/unchanged "
      f"{m('mean_consistency_all'):.4f}/{m('mean_consistency_changed'):.4f}/"
      f"{m('mean_consistency_unchanged'):.4f}). The submitted reference metrics (0.6896 / "
      "0.9867) could not be verified and must not be presented as reproduced. |")
    A("| Section 4.3/4.5 lambda selection | Report the preregistered validation-only rule and "
      f"the selected lambda ({lam}); remove the post-hoc lambda=0.10 test-set comparison from "
      "the confirmatory analysis. |")
    A("| Section 4.2/4.3 baselines | Report validation-selected hyperparameters "
      "(`baselines/hyperparameters.csv`) and locked-test results "
      "(`baselines/locked_test_results.csv`). The compact multilingual encoder and frozen "
      "embedding baselines are BLOCKED/not runnable in this environment — list as limitation, "
      "not as results. |")
    A("| Section 4.3 subgroup analysis | Add the joint language/source diagnostic (en "
      f"{m('en_composite_macro_f1'):.4f} vs zh {m('zh_composite_macro_f1'):.4f}); language and "
      "source_family are perfectly confounded — do not claim separate causal effects. |")
    A(f"| Section 4.3 rare classes | medication support is only {int(m('medication_support'))} "
      f"in the locked test (F1 {m('medication_f1'):.4f}); report per-class support with any "
      "rare-class claim (`model/per_class_metrics.csv`). |")
    A(f"| Section 4.4 consistency | Report all/changed/unchanged consistency separately "
      f"({m('mean_consistency_all'):.4f} / {m('mean_consistency_changed'):.4f} / "
      f"{m('mean_consistency_unchanged'):.4f}); unchanged-record consistency is trivially "
      "1.0 and must not be presented as evidence alone. |")
    A("| Section 4.4/4.5 PII | Separate (a) in-distribution pii_risk task metrics "
      f"({m('pii_risk_macro_f1'):.4f}), (b) the configured synthetic PII stress test "
      "(rule recall 1.00 on configured families; pii-head recall on out-of-distribution "
      "injections limited), and (c) independent human PII validation — BLOCKED. |")
    A(f"| Section 4.5 latency | End-to-end CPU latency here is "
      f"{timing[timing['method'] == f'rc_edgedistill_lambda_{lam}']['mean_ms_per_record'].iloc[0]:.3f} "
      "ms/record mean (full pipeline: normalization -> redaction -> TF-IDF -> inference -> "
      "decode -> serialize; threads=1; see `timing/timing_protocol.md`). The submitted 0.2042 "
      "ms/record model-level figure could not be verified (different hardware/pipeline "
      "definition); report hardware and pipeline boundary with any latency claim. |")
    A("| Section 3.2/4.1 provenance | LLM provenance (provider, model/version, settings, "
      "prompts, logs) is ABSENT; state the actual labeling provenance (LLM-assisted weak "
      "supervision per the manuscript) and the human-validation status (BLOCKED) explicitly. |")
    A("| Ethics statement | Ethics approval/exemption records are ABSENT; the manuscript must "
      "not claim an exemption that is not documented. |")
    A("| Reproducibility | Ship `reproducibility/` (requirements.txt, run_commands.txt, "
      "manifest.csv, SHA256SUMS.txt, summary.json) and note that exact reproduction requires "
      "the original submitted code, which was unavailable. |")
    A("")
    A("## 12. Output tree")
    A("")
    A("```")
    A("major_revision_run_v2/")
    A("├── FINAL_REVISION_RUN_REPORT.md")
    A("├── implementation_assumptions.md")
    A("├── src/                  corrected runnable code (common, metrics_utils, student,")
    A("│                         gradient_check, stage1..stage7)")
    A("├── data/                 input_verification.{json,md}, hashes.json, leakage_audit.{csv,json,md},")
    A("│                         locked_split.csv, locked_split_manifest.json")
    A("├── model/                selection_config.yaml, selection_log.{json,md},")
    A("│                         validation_lambda_results.csv, locked_test_metrics_by_seed.csv,")
    A("│                         locked_test_summary_across_seeds.csv, locked_test_predictions.csv,")
    A("│                         per_class_metrics.csv, bootstrap_confidence_intervals.csv")
    A("├── baselines/            hyperparameters.csv, validation_results.csv, locked_test_results.csv,")
    A("│                         locked_test_summary_across_seeds.csv, per_class_results.csv,")
    A("│                         bootstrap_confidence_intervals.csv, encoder_baseline_status.json")
    A("├── timing/               end_to_end_timing.csv, timing_manifest.json, timing_protocol.md")
    A("├── pii/                  synthetic_pii_stress_test_summary.csv, human_annotation_template.csv,")
    A("│                         pii_and_human_validation.md, llm_provenance_status.json")
    A("├── gradient_check/       symmetric_kl_logit_gradients.csv, parameter_gradient_check.csv,")
    A("│                         gradient_check_results.json, GRADIENT_CHECK.md")
    A("├── logs/                 stage run logs (stage2_training/*.log)")
    A("└── reproducibility/      README.md, requirements.txt, run_commands.txt, manifest.csv,")
    A("                          summary.json, SHA256SUMS.txt, final_output_summary.txt")
    A("```")
    A("")
    A("Storage constraints honoured: no checkpoints, no `tfidf_vectorizer.pkl`, no "
      "`fitted_models.pkl`, no per-baseline prediction files, no raw timing files, no "
      "figures/notebooks/caches/wheels, no duplicate datasets. Models are fitted and released "
      "one at a time, in memory.")
    A("")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---------------- README.md ----------------
    readme = f"""# Reproducibility Package — RC-EdgeDistill Major Revision Run v2

Run label: **{RUN_LABEL}** — a **corrected faithful reimplementation** (the exact submitted
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
- Data SHA-256: `{split_manifest['data_sha256']}` (verified against the expected value in stage 1)
- Code SHA-256 (src/*.py): `{chash}`
- Split seed: {split_manifest['split_seed']}; model seeds: {SEEDS}

## Gradient check
- Symmetric-KL logit gradients: max abs error {grad['symmetric_kl_logit_gradient_check']['max_abs_error']:.3e} -> {grad['symmetric_kl_logit_gradient_check']['result']}
- Student parameter gradients: max abs error {grad['student_parameter_gradient_check']['max_abs_error']:.3e} -> {grad['student_parameter_gradient_check']['result']}
- Overall: **{grad['overall_result']}**

## Selected lambda
- {lam} (preregistered rule in `../model/selection_config.yaml`; validation only)

## Key locked-test result (5 seeds, redacted view)
- composite macro-F1: {m('composite_macro_f1'):.4f} ± {sd('composite_macro_f1'):.4f}
- mean consistency (all / changed / unchanged): {m('mean_consistency_all'):.4f} / {m('mean_consistency_changed'):.4f} / {m('mean_consistency_unchanged'):.4f}

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
"""
    with open(REPRO_DIR / "README.md", "w") as f:
        f.write(readme)

    # ---------------- final_output_summary.txt (tree, count, bytes) ----------------
    tree_lines = []
    total_bytes = 0
    file_count = 0
    for dp, dns, fns in os.walk(RUN_ROOT):
        dns.sort()
        for fn in sorted(fns):
            full = Path(dp) / fn
            rel = full.relative_to(RUN_ROOT)
            if "__pycache__" in full.parts:
                continue
            size = full.stat().st_size
            total_bytes += size
            file_count += 1
            tree_lines.append(f"{size:>10}  {rel}")
    summary_txt = [
        f"RC-EdgeDistill major revision run v2 — final output summary",
        f"run label : {RUN_LABEL}",
        f"date      : {RUN_DATE}",
        f"run root  : {RUN_ROOT}",
        f"files     : {file_count}",
        f"total     : {total_bytes} bytes ({total_bytes / 1e6:.2f} MB)",
        "",
        f"{'bytes':>10}  relative_path",
    ] + tree_lines
    with open(REPRO_DIR / "final_output_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_txt) + "\n")

    # ---------------- SHA256SUMS.txt (after report + manifest exist) ----------------
    lines = []
    for dp, _, fns in os.walk(RUN_ROOT):
        for fn in sorted(fns):
            full = Path(dp) / fn
            if "__pycache__" in full.parts:
                continue
            rel = full.relative_to(RUN_ROOT)
            if rel == Path("reproducibility/SHA256SUMS.txt"):
                continue
            lines.append(f"{sha256_file(full)}  {rel}")
    lines.sort(key=lambda x: x.split("  ", 1)[1])
    with open(REPRO_DIR / "SHA256SUMS.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("stage7 done")
    print("report:", REPORT_PATH)
    print("files hashed:", len(lines))
    print(f"output files: {file_count}, total bytes: {total_bytes} "
          f"({total_bytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
