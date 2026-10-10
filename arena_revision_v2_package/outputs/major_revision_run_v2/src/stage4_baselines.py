"""Stage 4: TF-IDF CPU baselines on the same group-aware locked split.

Baselines (run sequentially): TF-IDF logistic regression, LinearSVM, SGD with
modified Huber loss, ridge classification, Complement Naive Bayes.
- Same locked train/validation/test partitions as the student; same TF-IDF
  feature construction (refit deterministically on the locked train views).
- Trained on the redacted_text of the locked train split (submitted-script
  convention); evaluated on both views; metrics reported on the redacted view.
- Small parameter grids; parameters selected on the locked VALIDATION split
  only (mean composite macro-F1 across the 5 seeds); then evaluated once per
  seed on the locked test split.
- Storage rules: NO per-baseline prediction files, NO fitted_models.pkl; each
  model is fitted and released in memory.
- Compact multilingual encoder baseline: BLOCKED (torch / sentence-transformers
  are not installed in this environment; no substitute architecture is used).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, BASELINES_DIR, LABEL_SPACES, CONSISTENCY_TASKS,
                    SEEDS, BASELINE_SEED, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
                    RUN_LABEL, RUN_DATE, write_json, ensure_dirs)
from metrics_utils import macro_f1, per_class_metrics
from stage2_train_student import build_features

GRIDS = {
    "tfidf_logistic_balanced": {
        # scikit-learn >= 1.5: liblinear no longer implicitly wraps multiclass in
        # one-vs-rest; the explicit OneVsRestClassifier wrapper reproduces the
        # historical liblinear OvR behaviour (recorded in implementation_assumptions.md).
        "factory": lambda C: OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               solver="liblinear", C=C,
                               random_state=BASELINE_SEED)),
        "param": "C", "grid": [0.5, 1.0, 4.0]},
    "tfidf_linear_svc_balanced": {
        "factory": lambda C: LinearSVC(class_weight="balanced", C=C,
                                       random_state=BASELINE_SEED, max_iter=10000),
        "param": "C", "grid": [0.5, 1.0, 4.0]},
    "tfidf_sgd_modified_huber": {
        "factory": lambda a: SGDClassifier(loss="modified_huber", class_weight="balanced",
                                           alpha=a, random_state=BASELINE_SEED,
                                           max_iter=2000, tol=1e-4),
        "param": "alpha", "grid": [1e-5, 1e-4, 1e-3]},
    "tfidf_ridge_balanced": {
        "factory": lambda a: RidgeClassifier(class_weight="balanced", alpha=a),
        "param": "alpha", "grid": [0.5, 1.0, 5.0]},
    "tfidf_complement_nb": {
        "factory": lambda a: ComplementNB(alpha=a),
        "param": "alpha", "grid": [0.5, 1.0, 2.0]},
}
TASKS = ["service_domain", "sentiment", "priority"]


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_DIR / "locked_split.csv")
    feats, feature_dim, char_n, word_n = build_features(df)

    splits = {}
    for split in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == split].reset_index(drop=True)
        splits[split] = {"df": sub, "Xo": feats[split]["Xo"],
                         "Xr": feats[split]["Xr"], "n": len(sub)}

    y = {sp: {t: splits[sp]["df"][t].values for t in TASKS} for sp in splits}
    changed_test = (splits["test"]["df"]["original_text"].astype(str) !=
                    splits["test"]["df"]["redacted_text"].astype(str)).values

    def run_task(clf, task, sp):
        clf.fit(splits["train"]["Xr"], y["train"][task])
        return (clf.predict(splits[sp]["Xo"]), clf.predict(splits[sp]["Xr"]))

    def composite(sp, po, pr):
        f1s = {t: macro_f1(y[sp][t], pr[t], LABEL_SPACES[t]) for t in TASKS}
        comp = float(np.mean(list(f1s.values())))
        cons = float(np.mean([(po[t] == pr[t]).mean() for t in TASKS]))
        return comp, f1s, cons

    val_rows, test_rows, class_rows, boot_rows, hyper_rows = [], [], [], [], []
    for name, spec in GRIDS.items():
        param, grid = spec["param"], spec["grid"]
        print(f"--- baseline {name} (grid {param}={grid}) ---")
        # --- grid evaluation on validation across seeds ---
        grid_scores = {}
        for val in grid:
            per_seed = []
            for seed in SEEDS:
                po, pr = {}, {}
                for t in TASKS:
                    clf = spec["factory"](val)
                    if name == "tfidf_sgd_modified_huber":
                        clf.set_params(random_state=seed)
                    po[t], pr[t] = run_task(clf, t, "validation")
                comp, f1s, cons = composite("validation", po, pr)
                per_seed.append(comp)
                val_rows.append({"run_label": RUN_LABEL, "baseline": name, "seed": seed,
                                 param: val, "val_composite_macro_f1": comp,
                                 **{f"val_{t}_macro_f1": f1s[t] for t in TASKS},
                                 "val_mean_consistency": cons})
            grid_scores[val] = float(np.mean(per_seed))
        best_val = max(grid_scores, key=lambda v: (grid_scores[v], -v))
        hyper_rows.append({"run_label": RUN_LABEL, "baseline": name, "parameter": param,
                           "grid": str(grid), "selected_value": best_val,
                           "selection_metric": "mean validation composite macro-F1 across seeds",
                           "selected_val_composite_macro_f1": grid_scores[best_val],
                           "selection_data": "locked validation split only"})
        print(f"{name}: selected {param}={best_val} (val composite {grid_scores[best_val]:.4f})")

        # --- locked test, once per seed, with selected params (in memory only) ---
        for seed in SEEDS:
            po, pr = {}, {}
            for t in TASKS:
                clf = spec["factory"](best_val)
                if name == "tfidf_sgd_modified_huber":
                    clf.set_params(random_state=seed)
                po[t], pr[t] = run_task(clf, t, "test")
            del clf
            comp, f1s, cons = composite("test", po, pr)
            agree = {t: (po[t] == pr[t]) for t in TASKS}
            cons_changed = float(np.mean([agree[t][changed_test].mean() for t in TASKS]))
            cons_unchanged = float(np.mean([agree[t][~changed_test].mean() for t in TASKS]))
            sub = {}
            for lang in ["en", "zh"]:
                m = (splits["test"]["df"]["language"].values == lang)
                sub[lang] = float(np.mean([
                    macro_f1(y["test"][t][m], pr[t][m], LABEL_SPACES[t]) for t in TASKS]))
            med_support = int((splits["test"]["df"]["service_domain"].values == "medication").sum())
            med = [r for r in per_class_metrics(y["test"]["service_domain"], pr["service_domain"],
                                                LABEL_SPACES["service_domain"])
                   if r["class"] == "medication"][0]
            test_rows.append({"run_label": RUN_LABEL, "baseline": name, "seed": seed,
                              param: best_val, "test_rows": int(len(splits["test"]["df"])),
                              "composite_macro_f1": comp,
                              **{f"{t}_macro_f1": f1s[t] for t in TASKS},
                              "mean_consistency_all": cons,
                              "mean_consistency_changed": cons_changed,
                              "mean_consistency_unchanged": cons_unchanged,
                              "en_composite_macro_f1": sub["en"],
                              "zh_composite_macro_f1": sub["zh"],
                              "medication_support": med_support, "medication_f1": med["f1"]})
            for t in TASKS:
                for r in per_class_metrics(y["test"][t], pr[t], LABEL_SPACES[t]):
                    class_rows.append({"run_label": RUN_LABEL, "baseline": name,
                                       "seed": seed, "task": t, **r})
            # paired bootstrap CI
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            n = len(splits["test"]["df"])
            boot = {"composite_macro_f1": [], "mean_consistency_all": []}
            for t in TASKS:
                boot[f"{t}_macro_f1"] = []
            for _ in range(BOOTSTRAP_REPLICATES):
                idx = rng.integers(0, n, size=n)
                f1s_r = {t: macro_f1(y["test"][t][idx], pr[t][idx], LABEL_SPACES[t])
                         for t in TASKS}
                boot["composite_macro_f1"].append(float(np.mean(list(f1s_r.values()))))
                for t in TASKS:
                    boot[f"{t}_macro_f1"].append(f1s_r[t])
                boot["mean_consistency_all"].append(float(np.mean(
                    [(po[t][idx] == pr[t][idx]).mean() for t in TASKS])))
            for m, vals in boot.items():
                lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
                boot_rows.append({"run_label": RUN_LABEL, "baseline": name, "seed": seed,
                                  "metric": m, "ci_lower": float(lo), "ci_upper": float(hi),
                                  "replicates": BOOTSTRAP_REPLICATES,
                                  "method": "paired bootstrap over locked-test records (percentile 95%)"})

    pd.DataFrame(hyper_rows).to_csv(BASELINES_DIR / "hyperparameters.csv", index=False)
    pd.DataFrame(val_rows).to_csv(BASELINES_DIR / "validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(BASELINES_DIR / "locked_test_results.csv", index=False)
    pd.DataFrame(class_rows).to_csv(BASELINES_DIR / "per_class_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(BASELINES_DIR / "bootstrap_confidence_intervals.csv",
                                   index=False)

    tr = pd.DataFrame(test_rows)
    num = [c for c in tr.columns if c not in {"run_label", "baseline", "seed", "C", "alpha"}]
    summ = tr.groupby("baseline")[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(BASELINES_DIR / "locked_test_summary_across_seeds.csv",
                              index=False)

    write_json(BASELINES_DIR / "encoder_baseline_status.json", {
        "run_label": RUN_LABEL,
        "date": RUN_DATE,
        "compact_multilingual_encoder": {
            "status": "BLOCKED",
            "reason": "torch and sentence-transformers are not installed in this "
                      "environment (no GPU/TPU stack, CPU-only sandbox); installing a "
                      "fine-tunable multilingual encoder (e.g. multilingual-e5-small or "
                      "paraphrase-multilingual-MiniLM-L12-v2 fine-tuning) is not possible "
                      "within the storage/network constraints, and the task forbids "
                      "substitute architectures (no Qwen/LoRA/QLoRA/transformers/new backbone).",
            "substitute_architecture_used": False,
            "frozen_embedding_baselines": "also not runnable here: sentence-transformers "
                                          "is not installed; the submitted frozen-embedding "
                                          "numbers (MiniLM/E5/BGE) could not be verified and are "
                                          "not reproduced in this run",
        }})
    print(tr.groupby("baseline")["composite_macro_f1"].agg(["mean", "std"]).to_string())
    print("stage4 done")


if __name__ == "__main__":
    main()
