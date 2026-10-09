"""Stage 4: TF-IDF baselines on the same locked split.

Baselines (per submitted script uploads/run_strong_cpu_baselines.py):
TF-IDF logistic regression, LinearSVM, SGD modified Huber, ridge, Complement NB.
- Same fitted TF-IDF vectorizer as the student (loaded from checkpoint pickle).
- Trained on redacted_text of the locked train split (submitted-script
  convention); evaluated on both views.
- Small parameter grids; parameters selected on the locked VALIDATION split
  only (mean composite macro-F1 across seeds); then evaluated once per seed on
  the locked test split.
- Compact multilingual encoder baseline: BLOCKED (torch/sentence-transformers
  not installed; prior install failure OSError 28; ~1.4 GiB available RAM). No
  substitute architecture used.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, BASELINES_DIR, LABEL_SPACES,
                    CONSISTENCY_TASKS, SEEDS, BASELINE_SEED,
                    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, RUN_LABEL, write_json)
from metrics_utils import macro_f1, per_class_metrics, idx_to_labels

CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
PRED_DIR = os.path.join(BASELINES_DIR, "predictions")

GRIDS = {
    "tfidf_logistic_balanced": {
        "factory": lambda C: LogisticRegression(max_iter=1000, class_weight="balanced",
                                                solver="liblinear", C=C,
                                                random_state=BASELINE_SEED),
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
    os.makedirs(PRED_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "locked_split.csv"))
    with open(os.path.join(CKPT_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vecs = pickle.load(f)
    char, word = vecs["char"], vecs["word"]

    def transform(texts):
        return hstack([char.transform(texts.astype(str)),
                       word.transform(texts.astype(str))]).astype(np.float32).tocsr()

    splits = {}
    for split in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == split].reset_index(drop=True)
        splits[split] = {"df": sub, "Xr": transform(sub["redacted_text"]),
                         "Xo": transform(sub["original_text"])}

    y = {sp: {t: splits[sp]["df"][t].values for t in TASKS}
         for sp in splits}
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

    val_rows, test_rows, class_rows, boot_rows, pred_rows = [], [], [], [], []
    hyper_rows = []
    fitted_best = {}
    for name, spec in GRIDS.items():
        param, grid = spec["param"], spec["grid"]
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

        # --- locked test, once per seed, with selected params ---
        fitted_best[name] = []
        for seed in SEEDS:
            po, pr, models = {}, {}, {}
            for t in TASKS:
                clf = spec["factory"](best_val)
                if name == "tfidf_sgd_modified_huber":
                    clf.set_params(random_state=seed)
                po[t], pr[t] = run_task(clf, t, "test")
                models[t] = clf
            fitted_best[name].append(models)
            comp, f1s, cons = composite("test", po, pr)
            agree = {t: (po[t] == pr[t]) for t in TASKS}
            cons_changed = float(np.mean([agree[t][changed_test].mean() for t in TASKS]))
            cons_unchanged = float(np.mean([agree[t][~changed_test].mean() for t in TASKS]))
            # subgroups (joint language/source diagnostic)
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
                    class_rows.append({"run_label": RUN_LABEL, "baseline": name, "seed": seed,
                                       "task": t, **r})
            pf = pd.DataFrame({"record_id": splits["test"]["df"]["record_id"], "seed": seed,
                               "language": splits["test"]["df"]["language"],
                               "source_family": splits["test"]["df"]["source_family"],
                               "changed_text": changed_test})
            for t in TASKS:
                pf[f"{t}_label"] = y["test"][t]
                pf[f"pred_{t}_original_view"] = po[t]
                pf[f"pred_{t}_redacted_view"] = pr[t]
            pf.to_csv(os.path.join(PRED_DIR, f"{name}_seed_{seed}.csv"), index=False)
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

    with open(os.path.join(BASELINES_DIR, "fitted_models.pkl"), "wb") as f:
        pickle.dump(fitted_best, f)
    pd.DataFrame(hyper_rows).to_csv(os.path.join(BASELINES_DIR, "hyperparameters.csv"), index=False)
    pd.DataFrame(val_rows).to_csv(os.path.join(BASELINES_DIR, "validation_results.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(BASELINES_DIR, "locked_test_results.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(os.path.join(BASELINES_DIR, "per_class_results.csv"), index=False)
    pd.DataFrame(boot_rows).to_csv(os.path.join(BASELINES_DIR, "bootstrap_confidence_intervals.csv"), index=False)

    # mean/std across seeds per baseline
    tr = pd.DataFrame(test_rows)
    num = [c for c in tr.columns if c not in {"run_label", "baseline", "seed", "C", "alpha"}]
    summ = tr.groupby("baseline")[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(BASELINES_DIR, "locked_test_summary_across_seeds.csv"), index=False)

    write_json(os.path.join(BASELINES_DIR, "encoder_baseline_status.json"), {
        "run_label": RUN_LABEL,
        "compact_multilingual_encoder": {
            "status": "BLOCKED",
            "reason": "torch and sentence-transformers are not installed in this environment; "
                      "the earlier required install failed with "
                      "pip._vendor.urllib3.exceptions.ProtocolError: (\"Connection broken: "
                      "OSError(28, 'No space left on device')\", OSError(28, 'No space left on device')); "
                      "available RAM (~1.4 GiB) is marginal for fine-tuning a multilingual encoder "
                      "(e.g., multilingual-e5-small ~470 MB fp32 weights plus activations).",
            "substitute_architecture_used": False,
        }})
    print(tr.groupby("baseline")["composite_macro_f1"].agg(["mean", "std"]).to_string())
    print("stage4 done")


if __name__ == "__main__":
    main()
