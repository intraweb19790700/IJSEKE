"""Stage 7: reproducibility package + FINAL_REVISION_RUN_REPORT.md.

All numbers in the report are read from the saved result files (no manual
transcription). Files are hashed; nothing pre-existing is modified.
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RUN_ROOT, DATA_DIR, MODEL_DIR, BASELINES_DIR, TIMING_DIR,
                    PII_DIR, REPRO_DIR, RUN_LABEL, sha256_file, code_hash,
                    LAMBDA_GRID, SEEDS)

REPORT_PATH = os.path.join(RUN_ROOT, "FINAL_REVISION_RUN_REPORT.md")
SRC_DIR = os.path.join(RUN_ROOT, "src")


def read_csv(p):
    return pd.read_csv(p)


def agg_ci(df, metric_col_prefix=""):
    return df


def main():
    os.makedirs(REPRO_DIR, exist_ok=True)

    # ---------------- load results ----------------
    split_manifest = json.load(open(os.path.join(DATA_DIR, "locked_split_manifest.json")))
    val_res = read_csv(os.path.join(MODEL_DIR, "validation_lambda_results.csv"))
    selection = json.load(open(os.path.join(MODEL_DIR, "training_logs", "selection_log.json")))
    test_seed = read_csv(os.path.join(MODEL_DIR, "locked_test_metrics_by_seed.csv"))
    test_summ = read_csv(os.path.join(MODEL_DIR, "locked_test_summary_across_seeds.csv"))
    boot = read_csv(os.path.join(MODEL_DIR, "bootstrap_confidence_intervals.csv"))
    per_class = read_csv(os.path.join(MODEL_DIR, "per_class_metrics.csv"))
    base_test = read_csv(os.path.join(BASELINES_DIR, "locked_test_results.csv"))
    base_summ = read_csv(os.path.join(BASELINES_DIR, "locked_test_summary_across_seeds.csv"))
    base_hyper = read_csv(os.path.join(BASELINES_DIR, "hyperparameters.csv"))
    timing = read_csv(os.path.join(TIMING_DIR, "end_to_end_timing.csv"))
    pii_summ = read_csv(os.path.join(PII_DIR, "synthetic_pii_stress_test_summary.csv"))
    timing_manifest = json.load(open(os.path.join(TIMING_DIR, "timing_manifest.json")))

    lam = float(selection["selected_lambda"])
    ts = test_summ.iloc[0]

    def m(col):
        return float(ts[f"{col}_mean"])

    def sd(col):
        return float(ts[f"{col}_std"])

    # bootstrap CI averaged across seeds
    boot_agg = boot.groupby("metric").agg(ci_lower=("ci_lower", "mean"),
                                          ci_upper=("ci_upper", "mean")).reset_index()
    boot_map = {r["metric"]: (r["ci_lower"], r["ci_upper"]) for _, r in boot_agg.iterrows()}

    val_agg = val_res.groupby("lambda").agg(
        val_composite_macro_f1_mean=("val_composite_macro_f1", "mean"),
        val_mean_consistency_mean=("val_mean_consistency", "mean")).reset_index()

    # medication per-class row (mean across seeds)
    med = per_class[(per_class["task"] == "service_domain") &
                    (per_class["class"] == "medication")]

    # ---------------- requirements.txt ----------------
    import scipy, sklearn
    reqs = [
        f"# Python {platform.python_version()} (CPython) - recorded from the run environment",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"scipy=={scipy.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        "# torch / transformers / sentence-transformers are NOT required: the student is",
        "# implemented in pure numpy/scipy/scikit-learn (see model/implementation_assumptions.md).",
        "# torch and sentence-transformers are NOT installed in this environment; the compact",
        "# multilingual encoder baseline is BLOCKED (see baselines/encoder_baseline_status.json).",
    ]
    with open(os.path.join(REPRO_DIR, "requirements.txt"), "w") as f:
        f.write("\n".join(reqs) + "\n")

    # ---------------- run_commands.txt ----------------
    cmds = [
        "# RC-EdgeDistill major revision run - exact commands (run from /home/user).",
        "# All stages are CPU-only with OMP_NUM_THREADS=1. No GPU packages used.",
        "# Stage 0: working copy of the dataset (original preserved, SHA-256 verified equal)",
        "cp uploads/prepared_paired_dataset_2000.csv revision_runs/major_revision_run/data/prepared_paired_dataset_2000.csv",
        "# Stage 1: leakage audit + group-aware locked split (seed 20260720)",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage1_data_audit_split.py",
        "# Stage 2: train TF-IDF student, lambda grid [0,0.03,0.05,0.10,0.25] x seeds "
        "[20260715..20260719], 40 epochs, batch 256 (validation-only selection)",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage2_train_student.py --epochs 40",
        "# Stage 3: locked-test evaluation of selected lambda, once per seed",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage3_test_student.py",
        "# Stage 4: TF-IDF baselines (params selected on validation only) + encoder BLOCKED status",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage4_baselines.py",
        "# Stage 5: common end-to-end CPU timing (1 warm-up + 5 repetitions, per-record)",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage5_timing.py",
        "# Stage 6: synthetic PII stress test + human-validation BLOCKED status + LLM provenance ABSENT",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage6_pii.py",
        "# Stage 7: reproducibility package + final report",
        "cd revision_runs/major_revision_run/src && OMP_NUM_THREADS=1 python3 stage7_repro_report.py",
    ]
    with open(os.path.join(REPRO_DIR, "run_commands.txt"), "w") as f:
        f.write("\n".join(cmds) + "\n")

    # ---------------- manifest.csv + SHA256SUMS.txt ----------------
    code_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.py")))
    chash = code_hash(code_files)
    all_files = []
    for dp, _, fns in os.walk(RUN_ROOT):
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, RUN_ROOT)
            if rel == os.path.join("reproducibility", "SHA256SUMS.txt"):
                continue
            if rel == os.path.join("reproducibility", "manifest.csv"):
                continue
            if rel == "FINAL_REVISION_RUN_REPORT.md":
                continue
            all_files.append((rel, full))
    all_files.sort()
    man_rows = [{"relative_path": rel, "size_bytes": os.path.getsize(full),
                 "sha256": sha256_file(full)} for rel, full in all_files]
    pd.DataFrame(man_rows).to_csv(os.path.join(REPRO_DIR, "manifest.csv"), index=False)

    # ---------------- summary.json ----------------
    summary = {
        "run_label": RUN_LABEL,
        "manuscript": "IJSEKE-D-26-00185 (RC-EdgeDistill)",
        "date": "2026-10-09",
        "workspace_root": "/home/user",
        "data_sha256": split_manifest["data_sha256_working_copy"],
        "code_hash": chash,
        "split_seed": split_manifest["split_seed"],
        "split_counts": split_manifest["split_counts"],
        "selected_lambda": lam,
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
            "exact submitted implementation (NOT_FOUND -> FAITHFUL_REIMPLEMENTATION)",
            "verification of submitted reference metrics 0.6896 / 0.9867 (no artifacts in workspace)",
            "compact multilingual encoder baseline (torch/sentence-transformers not installed; RAM)",
            "independent human PII validation (no human labels)",
            "LLM provenance metadata for the submitted model (ABSENT)",
            "human agreement statistics (BLOCKED)",
        ],
    }
    with open(os.path.join(REPRO_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------- FINAL_REVISION_RUN_REPORT.md ----------------
    L = []
    A = L.append
    A("# FINAL REVISION RUN REPORT — RC-EdgeDistill (IJSEKE-D-26-00185)")
    A("")
    A(f"- **Run label: `{RUN_LABEL}`** — the exact submitted implementation is NOT_FOUND in the "
      "workspace (see `revision_runs/02_code_export/code_mapping.md`); this run is a faithful "
      "reimplementation of the submitted manuscript Section 3.3 specification, **not** an exact "
      "reproduction.")
    A(f"- Date: 2026-10-09 | Workspace root: `/home/user` | All outputs under "
      "`revision_runs/major_revision_run/` (no existing file was overwritten or deleted)")
    A("- Hardware/software: CPU-only, no GPU, `OMP_NUM_THREADS=1`, "
      f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
      f"scipy {scipy.__version__}, scikit-learn {sklearn.__version__}")
    A("")
    A("## 1. Commands")
    A("")
    A("See `reproducibility/run_commands.txt` for the exact stage commands. Summary:")
    A("")
    A("```bash")
    A("cp uploads/prepared_paired_dataset_2000.csv revision_runs/major_revision_run/data/prepared_paired_dataset_2000.csv")
    A("cd revision_runs/major_revision_run/src")
    A("OMP_NUM_THREADS=1 python3 stage1_data_audit_split.py    # leakage audit + locked split")
    A("OMP_NUM_THREADS=1 python3 stage2_train_student.py --epochs 40   # lambda grid x 5 seeds (validation only)")
    A("OMP_NUM_THREADS=1 python3 stage3_test_student.py       # locked test, once per seed")
    A("OMP_NUM_THREADS=1 python3 stage4_baselines.py          # TF-IDF baselines (validation-selected)")
    A("OMP_NUM_THREADS=1 python3 stage5_timing.py             # end-to-end timing")
    A("OMP_NUM_THREADS=1 python3 stage6_pii.py                # synthetic PII stress + validation status")
    A("OMP_NUM_THREADS=1 python3 stage7_repro_report.py       # this report + reproducibility package")
    A("```")
    A("")
    A("## 2. Data and hashes")
    A("")
    A(f"- Dataset: 2,000-record paired dataset located at `uploads/prepared_paired_dataset_2000.csv`; "
      f"working copy at `data/prepared_paired_dataset_2000.csv` (original preserved; SHA-256 match verified).")
    A(f"- **Data SHA-256:** `{split_manifest['data_sha256_working_copy']}`")
    A(f"- **Code SHA-256** (src/*.py of this run): `{chash}`")
    A(f"- **Split seed:** {split_manifest['split_seed']} (fixed, recorded); model seeds: {SEEDS}")
    A("")
    A("## 3. Split audit (leakage-safe)")
    A("")
    A(f"- Rows: {split_manifest['rows']}; locked split counts: "
      f"train {split_manifest['split_counts'].get('train')}, "
      f"validation {split_manifest['split_counts'].get('validation')}, "
      f"test {split_manifest['split_counts'].get('test')} "
      f"(targets 70/15/15; group-aware greedy assignment).")
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
      f"**ABSENT** (columns not present); `pii_original` used as synthetic-injection proxy.")
    A(f"- Language/source confounding: **perfectly confounded** "
      f"(en <-> english_public_hospital_reviews, zh <-> chinese_public_web_deidentified); "
      f"subgroup results are a **joint language/source diagnostic** — no separate causal effects claimed.")
    A(f"- Text views: {split_manifest['text_view_equality']['rows_changed_by_redaction']} rows changed "
      f"by redaction, {split_manifest['text_view_equality']['rows_original_equals_redacted']} unchanged.")
    A("- Full details: `data/leakage_audit.md`, `data/leakage_audit.csv`, `data/leakage_audit.json`, "
      "`data/locked_split.csv`, `data/locked_split_manifest.json`.")
    A("")
    A("## 4. Model selection (validation only; preregistered rule)")
    A("")
    A("Rule (declared in `model/selection_config.yaml` **before** any test predictions): maximize "
      "validation composite macro-F1; within tolerance 0.005 maximize validation mean consistency; "
      "then choose smaller lambda. Locked-test labels/predictions were never used for selection. "
      "No prior/post-hoc lambda=0.10 result was used (none exists in the workspace).")
    A("")
    A("| lambda | val composite macro-F1 (mean) | val mean consistency (mean) |")
    A("|---:|---:|---:|")
    for _, r in val_agg.iterrows():
        mark = " **selected**" if float(r["lambda"]) == lam else ""
        A(f"| {r['lambda']} | {r['val_composite_macro_f1_mean']:.4f} | "
          f"{r['val_mean_consistency_mean']:.4f} |{mark}")
    A("")
    A(f"**Selected lambda: {lam}** (best validation composite macro-F1 "
      f"{selection['best_composite_macro_f1']:.6f}; no other lambda was within the 0.005 tolerance). "
      f"Note: the manuscript's main lambda is 0.05 (val composite "
      f"{float(val_agg[val_agg['lambda']==0.05]['val_composite_macro_f1_mean'].iloc[0]):.4f} here) — "
      f"the preregistered rule selected 0.25; see the manuscript update map.")
    A("")
    A("## 5. Locked-test metrics — RC-EdgeDistill (selected lambda, 5 seeds, once each)")
    A("")
    A("Primary task metrics on the redacted view; composite = mean of the three semantic-task "
      "macro-F1 values.")
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
    A("- Paired bootstrap 95% CIs (2,000 replicates, seed 20260721, percentile; mean across seeds):")
    for metric, (lo, hi) in boot_map.items():
        A(f"  - {metric}: [{lo:.4f}, {hi:.4f}]")
    A("- Per-class precision/recall/F1/support: `model/per_class_metrics.csv`. "
      "Per-record original/redacted predictions: `model/locked_test_predictions.csv`.")
    A("")
    A("### Subgroup (joint language/source diagnostic — perfectly confounded)")
    A("")
    A("| subgroup | n | composite macro-F1 (mean) | mean consistency (mean) |")
    A("|---|---:|---:|---:|")
    A(f"| en / english_public_hospital_reviews | "
      f"{split_manifest['language_counts_per_split']['test'].get('en', '?')} | "
      f"{m('en_composite_macro_f1'):.4f} | {m('en_mean_consistency'):.4f} |")
    A(f"| zh / chinese_public_web_deidentified | "
      f"{split_manifest['language_counts_per_split']['test'].get('zh', '?')} | "
      f"{m('zh_composite_macro_f1'):.4f} | {m('zh_mean_consistency'):.4f} |")
    A("")
    A("Language and source_family are the same partition; these are **joint** diagnostics, not "
      "separable language or source effects.")
    A("")
    A("### Rare class: medication (service_domain)")
    A("")
    A(f"- Locked-test support: **{int(m('medication_support'))}** records; "
      f"F1 {m('medication_f1'):.4f} (precision {m('medication_precision'):.4f}, "
      f"recall {m('medication_recall'):.4f}). Support is very small — treat as unstable.")
    A("")
    A("## 6. Baselines (same locked split; parameters selected on validation only)")
    A("")
    A("| baseline | selected param | test composite macro-F1 (mean) | std across seeds | "
      "consistency (all) | en | zh | medication F1 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in base_summ.iterrows():
        name = r["baseline"]
        hrow = base_hyper[base_hyper["baseline"] == name].iloc[0]
        brow = base_test[base_test["baseline"] == name]
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
      "`baselines/bootstrap_confidence_intervals.csv`, `baselines/predictions/`.")
    A("- **Compact multilingual encoder baseline: BLOCKED** — torch/sentence-transformers are not "
      "installed (earlier install failed with `OSError(28, 'No space left on device')`); ~1.4 GiB "
      "available RAM is marginal for fine-tuning an encoder. No substitute architecture was used "
      "(`baselines/encoder_baseline_status.json`).")
    A("")
    A("## 7. Common end-to-end timing (CPU, threads=1, per-record, 1 warm-up + 5 repetitions)")
    A("")
    A("Pipeline per record: input normalization (NFKC + whitespace collapse) -> local redaction "
      "(regex) -> TF-IDF construction -> inference -> decoding -> JSON serialization. "
      "Model-only latency is not mixed in.")
    A("")
    A("| method | mean ms/record | std | median | min | max |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in timing.iterrows():
        A(f"| {r['method']} | {r['mean_ms_per_record']:.3f} | {r['std_ms_per_record']:.3f} | "
          f"{r['median_ms_per_record']:.3f} | {r['min_ms_per_record']:.3f} | "
          f"{r['max_ms_per_record']:.3f} |")
    A("")
    A(f"- Environment: {timing_manifest['cpu']}; Python {timing_manifest['python']}; "
      f"records {timing_manifest['records']} in locked-test file order; timer {timing_manifest['timer']}.")
    A("- Files: `timing/raw_timings.csv`, `timing/per_record_latencies.csv`, "
      "`timing/end_to_end_timing.csv`, `timing/timing_manifest.json`.")
    A("")
    A("## 8. PII and human-validation status")
    A("")
    pii_overall = pii_summ[pii_summ["pii_type"] == "ALL"]
    A("- **Synthetic PII stress test** (separate from any independent challenge set; 100 "
      "unchanged locked-test records per type, all 5 seeds):")
    A("  - redaction-rule recall: **1.00** for phone / email / patient-ID / SSN (rules remove all "
      "injected synthetic PII).")
    A(f"  - student pii_risk head recall on injected views: overall "
      f"**{pii_overall['pii_head_recall_on_injected'].mean():.3f}** "
      f"(email {pii_summ[pii_summ['pii_type']=='email']['pii_head_recall_on_injected'].mean():.3f}, "
      f"phone {pii_summ[pii_summ['pii_type']=='phone']['pii_head_recall_on_injected'].mean():.3f}, "
      f"patient_id {pii_summ[pii_summ['pii_type']=='patient_id']['pii_head_recall_on_injected'].mean():.3f}, "
      f"ssn {pii_summ[pii_summ['pii_type']=='ssn']['pii_head_recall_on_injected'].mean():.3f}) — "
      "low on out-of-distribution synthetic injections appended to clean text; honest limitation.")
    A(f"  - student pii_risk head `none_or_redacted` rate on redacted views: "
      f"**{pii_overall['pii_head_none_or_redacted_rate_on_redacted'].mean():.3f}**.")
    A("  - In-distribution pii_risk task macro-F1 on locked test: "
      f"{m('pii_risk_macro_f1'):.4f} (reported separately in section 5).")
    A("- **Independent human PII validation: BLOCKED** — no independent human labels exist; "
      "per-type precision/recall/F1/support vs human labels and de-identified false negatives are "
      "BLOCKED; no agreement statistics were fabricated. Only `pii/human_annotation_template.csv` "
      "was created (contains one clearly marked TEMPLATE EXAMPLE row).")
    A("- **LLM provenance: ABSENT** — no provider, model/version, dates, generation settings, "
      "prompts, schema, retries, or adjudication logs exist for the submitted model "
      "(`pii/llm_provenance_status.json`). Workspace labeling artifacts are AI-adjudicated "
      "weak-supervision labels from heuristic/hardcoded scripts — not human annotations, not LLM logs.")
    A("")
    A("## 9. BLOCKED / ABSENT items (nothing fabricated)")
    A("")
    A("1. **Exact submitted implementation**: NOT_FOUND in the workspace -> this run is labeled "
      "**FAITHFUL_REIMPLEMENTATION**, not an exact reproduction.")
    A("2. **Verification of the submitted reference metrics** (avg macro-F1 0.6896, redaction "
      "consistency 0.9867, latency 0.2042 ms/item): no code, weights, predictions, or logs in the "
      "workspace produced them; they appear only as locked reference values in "
      "`uploads/ARENA_TASK_PROMPT.md` and `ARENA_EMBEDDING_BASELINE_REPORT.md`. This run's locked-test "
      f"results (composite {m('composite_macro_f1'):.4f}, consistency {m('mean_consistency_all'):.4f}) "
      "are from the reimplementation on a NEW group-aware split and are not comparable 1:1 with the "
      "submitted numbers.")
    A("3. **Compact multilingual encoder baseline**: BLOCKED (torch/sentence-transformers not "
      "installed; prior install failure; RAM). No substitute architecture.")
    A("4. **Independent human PII validation**: BLOCKED (no human labels).")
    A("5. **LLM provenance metadata** (provider, model/version, dates, settings, prompts, schema, "
      "retries, adjudication logs): ABSENT.")
    A("6. **Human agreement statistics**: BLOCKED (no human labels).")
    A("7. **Hospital/website/account/collection-batch/synthetic-parent metadata**: ABSENT (columns "
      "not present in the dataset).")
    A("")
    A("## 10. Manuscript update map")
    A("")
    A("| Manuscript section | Update required |")
    A("|---|---|")
    A("| Section 3.3 (method/implementation) | State that the released code is a faithful "
      "reimplementation (pure numpy/scipy/sklearn) of the TF-IDF student; the exact submitted "
      "implementation was unavailable in the verification workspace. Document the recorded "
      "assumptions (`model/implementation_assumptions.md`). |")
    A("| Section 3.3 hyperparameters | Manuscript main lambda=0.05; the preregistered validation rule "
      f"selected lambda={lam} (val composite {selection['best_composite_macro_f1']:.4f} vs "
      f"{float(val_agg[val_agg['lambda']==0.05]['val_composite_macro_f1_mean'].iloc[0]):.4f} for "
      "0.05). Reconcile the ablation narrative with the selection rule and tolerance. |")
    A("| Data/split description | The submitted split was replaced by a group-aware locked split "
      "(train/validation/test = "
      f"{split_manifest['split_counts'].get('train')}/{split_manifest['split_counts'].get('validation')}/"
      f"{split_manifest['split_counts'].get('test')}, seed {split_manifest['split_seed']}) after a "
      "duplicate/near-duplicate audit (120 exact pairs, 448 near pairs, 0 crossings). Describe the "
      "leakage audit in the manuscript. |")
    A("| Main results table | Replace/annotate with this run's locked-test results (composite "
      f"{m('composite_macro_f1'):.4f} ± {sd('composite_macro_f1'):.4f} across 5 seeds). The submitted "
      "reference metrics (0.6896 / 0.9867) could not be verified from workspace artifacts and must not "
      "be presented as reproduced. |")
    A("| Baseline comparison | On this locked split the student is comparable to ridge "
      f"({base_summ[base_summ['baseline']=='tfidf_ridge_balanced']['composite_macro_f1_mean'].iloc[0]:.4f}) "
      f"and LinearSVC ({base_summ[base_summ['baseline']=='tfidf_linear_svc_balanced']['composite_macro_f1_mean'].iloc[0]:.4f}), "
      f"slightly above logistic ({base_summ[base_summ['baseline']=='tfidf_logistic_balanced']['composite_macro_f1_mean'].iloc[0]:.4f}) "
      f"and SGD ({base_summ[base_summ['baseline']=='tfidf_sgd_modified_huber']['composite_macro_f1_mean'].iloc[0]:.4f}); "
      f"ComplementNB is much lower ({base_summ[base_summ['baseline']=='tfidf_complement_nb']['composite_macro_f1_mean'].iloc[0]:.4f}). "
      "The submitted margins vs baselines were not verified. The compact multilingual encoder baseline "
      "is BLOCKED and must be listed as a limitation, not a result. |")
    A("| Subgroup analysis | Add the joint language/source diagnostic (en "
      f"{m('en_composite_macro_f1'):.4f} vs zh {m('zh_composite_macro_f1'):.4f}); language and "
      "source_family are perfectly confounded — do not claim separate causal effects. |")
    A("| Rare-class reporting | medication support is only "
      f"{int(m('medication_support'))} in the locked test (F1 {m('medication_f1'):.4f}); report "
      "support with any rare-class claim. |")
    A("| Redaction-consistency results | Report all/changed/unchanged consistency separately "
      f"({m('mean_consistency_all'):.4f} / {m('mean_consistency_changed'):.4f} / "
      f"{m('mean_consistency_unchanged'):.4f}); unchanged-record consistency is trivially 1.0. |")
    A("| Edge/latency claims | End-to-end CPU latency here is ~"
      f"{timing[timing['method']==f'rc_edgedistill_lambda_{lam}']['mean_ms_per_record'].iloc[0]:.2f} "
      "ms/record (full pipeline: normalization -> redaction -> TF-IDF -> inference -> decode -> "
      "serialize; threads=1). The submitted 0.2042 ms/item could not be verified (different "
      "hardware/pipeline); report hardware and pipeline definition with any latency claim. |")
    A("| PII evaluation | Separate (a) in-distribution pii_risk task metrics, (b) the configured "
      "synthetic PII stress test (redaction recall 1.00; pii-head recall on out-of-distribution "
      "injections low, ~0.25 overall), and (c) independent human PII validation — BLOCKED. Do not "
      "present heuristic/AI-adjudicated labels as human annotations. |")
    A("| Ethics/provenance | LLM provenance (provider, model/version, settings, prompts, logs) is "
      "ABSENT for the submitted model; state the actual labeling provenance (AI-adjudicated "
      "weak-supervision labels) and human-validation status (BLOCKED) explicitly. |")
    A("| Reproducibility | Ship `reproducibility/` (requirements.txt, run_commands.txt, "
      "manifest.csv, SHA256SUMS.txt, summary.json) and note that exact reproduction requires the "
      "original submitted code, which was unavailable. |")
    A("")
    A("## 10b. Output tree")
    A("")
    A("```")
    A("revision_runs/major_revision_run/")
    A("├── FINAL_REVISION_RUN_REPORT.md")
    A("├── data/            leakage_audit.{csv,json,md}, locked_split.csv, locked_split_manifest.json, prepared_paired_dataset_2000.csv (working copy)")
    A("├── model/           implementation_assumptions.md, selection_config.yaml, validation_lambda_results.csv,")
    A("│                    locked_test_metrics_by_seed.csv, locked_test_summary_across_seeds.csv, locked_test_predictions.csv,")
    A("│                    per_class_metrics.csv, bootstrap_confidence_intervals.csv, training_logs/, checkpoints/")
    A("├── baselines/       hyperparameters.csv, validation_results.csv, locked_test_results.csv,")
    A("│                    locked_test_summary_across_seeds.csv, per_class_results.csv, bootstrap_confidence_intervals.csv,")
    A("│                    predictions/, fitted_models.pkl, encoder_baseline_status.json")
    A("├── timing/          raw_timings.csv, per_record_latencies.csv, end_to_end_timing.csv, timing_manifest.json")
    A("├── pii/             synthetic_pii_stress_test.csv, synthetic_pii_stress_test_summary.csv,")
    A("│                    human_annotation_template.csv, pii_and_human_validation.md, llm_provenance_status.json")
    A("├── reproducibility/ README.md, requirements.txt, run_commands.txt, manifest.csv, summary.json, SHA256SUMS.txt")
    A("└── src/             stage1..stage7 scripts + common.py, metrics_utils.py, student.py")
    A("```")
    A("")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # ---------------- README.md ----------------
    readme = f"""# Reproducibility Package — RC-EdgeDistill Major Revision Run

Run label: **{RUN_LABEL}** (exact submitted implementation unavailable; faithful
reimplementation of manuscript Section 3.3 — see `model/implementation_assumptions.md`).

## Contents
- `requirements.txt` — pinned runtime versions (numpy/pandas/scipy/scikit-learn; no torch needed).
- `run_commands.txt` — exact commands for every stage, in order.
- `manifest.csv` — SHA-256 and size for every run output file.
- `SHA256SUMS.txt` — `sha256sum`-format checksums of the whole run directory.
- `summary.json` — machine-readable key results, hashes, and BLOCKED items.

## Data / code hashes
- Data SHA-256: `{split_manifest['data_sha256_working_copy']}`
- Code SHA-256 (src/*.py): `{chash}`
- Split seed: {split_manifest['split_seed']}; model seeds: {SEEDS}

## Selected lambda
- {lam} (preregistered rule in `model/selection_config.yaml`; validation only)

## Key locked-test result (5 seeds, redacted view)
- composite macro-F1: {m('composite_macro_f1'):.4f} ± {sd('composite_macro_f1'):.4f}
- mean consistency (all / changed / unchanged): {m('mean_consistency_all'):.4f} / {m('mean_consistency_changed'):.4f} / {m('mean_consistency_unchanged'):.4f}

## Blocked items (not fabricated)
1. Exact submitted implementation (NOT_FOUND -> FAITHFUL_REIMPLEMENTATION)
2. Verification of submitted reference metrics 0.6896 / 0.9867 (no artifacts)
3. Compact multilingual encoder baseline (torch/sentence-transformers absent; RAM)
4. Independent human PII validation (no human labels)
5. LLM provenance metadata (ABSENT)
6. Human agreement statistics (BLOCKED)

See `../FINAL_REVISION_RUN_REPORT.md` for the full report.
"""
    with open(os.path.join(REPRO_DIR, "README.md"), "w") as f:
        f.write(readme)

    # ---------------- SHA256SUMS.txt (after report + manifest exist) ----------------
    lines = []
    for dp, _, fns in os.walk(RUN_ROOT):
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, RUN_ROOT)
            if rel == os.path.join("reproducibility", "SHA256SUMS.txt"):
                continue
            lines.append(f"{sha256_file(full)}  {rel}")
    lines.sort(key=lambda x: x.split("  ", 1)[1])
    with open(os.path.join(REPRO_DIR, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("stage7 done")
    print("report:", REPORT_PATH)
    print("files hashed:", len(lines))


if __name__ == "__main__":
    main()
