"""Stage 3: locked-test evaluation of the selected lambda, one evaluation per
seed. Produces per-seed metrics, per-record original/redacted predictions,
per-class metrics, subgroup (joint language/source) results, medication
rare-class results, changed/unchanged consistency, and paired bootstrap 95% CIs.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, LABEL_SPACES, CONSISTENCY_TASKS,
                    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, RUN_LABEL, write_json)
from metrics_utils import (macro_f1, per_class_metrics, idx_to_labels,
                           bootstrap_ci)
from student import TfidfStudent

CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "locked_split.csv"))
    test = df[df["locked_split"] == "test"].reset_index(drop=True)
    with open(os.path.join(CKPT_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vecs = pickle.load(f)
    char, word = vecs["char"], vecs["word"]

    def transform(texts):
        return hstack([char.transform(texts.astype(str)),
                       word.transform(texts.astype(str))]).astype(np.float32).tocsr()

    Xo = transform(test["original_text"])
    Xr = transform(test["redacted_text"])
    feature_dim = len(char.vocabulary_) + len(word.vocabulary_)
    changed = (test["original_text"].astype(str) != test["redacted_text"].astype(str)).values

    sel = write_json  # noqa - keep import used
    import json
    with open(os.path.join(MODEL_DIR, "training_logs", "selection_log.json")) as f:
        selection = json.load(f)
    lam = float(selection["selected_lambda"])
    seeds = [int(s) for s in selection.get("seeds", [20260715, 20260716, 20260717, 20260718, 20260719])]
    print(f"selected lambda={lam}, evaluating seeds {seeds} once each on locked test")

    y = {}
    for task in LABEL_SPACES:
        if task == "pii_risk":
            y["pii_risk_original"] = test["pii_original"].map(LABEL_SPACES[task].index).values
            y["pii_risk_redacted"] = test["pii_redacted"].map(LABEL_SPACES[task].index).values
        else:
            y[task] = test[task].map(LABEL_SPACES[task].index).values

    metric_rows = []
    pred_frames = []
    class_rows = []
    boot_rows = []
    for seed in seeds:
        model = TfidfStudent(feature_dim, seed)
        model.load(os.path.join(CKPT_DIR, f"student_lambda_{lam}_seed_{seed}.npz"))
        po = model.predict_idx(Xo)
        pr = model.predict_idx(Xr)

        # primary metrics on redacted view
        f1_red = {}
        for task in LABEL_SPACES:
            tgt = y["pii_risk_redacted"] if task == "pii_risk" else y[task]
            f1_red[task] = macro_f1(idx_to_labels(task, tgt), idx_to_labels(task, pr[task]),
                                    LABEL_SPACES[task])
        comp = float(np.mean([f1_red[t] for t in CONSISTENCY_TASKS]))
        f1_orig = {}
        for task in LABEL_SPACES:
            tgt = y["pii_risk_original"] if task == "pii_risk" else y[task]
            f1_orig[task] = macro_f1(idx_to_labels(task, tgt), idx_to_labels(task, po[task]),
                                     LABEL_SPACES[task])
        comp_orig = float(np.mean([f1_orig[t] for t in CONSISTENCY_TASKS]))

        # consistency all / changed / unchanged
        cons = {}
        for t in CONSISTENCY_TASKS:
            agree = (po[t] == pr[t])
            cons[f"{t}_consistency_all"] = float(agree.mean())
            cons[f"{t}_consistency_changed"] = float(agree[changed].mean())
            cons[f"{t}_consistency_unchanged"] = float(agree[~changed].mean())
        mean_cons = float(np.mean([cons[f"{t}_consistency_all"] for t in CONSISTENCY_TASKS]))
        mean_cons_changed = float(np.mean([cons[f"{t}_consistency_changed"] for t in CONSISTENCY_TASKS]))
        mean_cons_unchanged = float(np.mean([cons[f"{t}_consistency_unchanged"] for t in CONSISTENCY_TASKS]))

        # joint language/source diagnostic (perfectly confounded partitions)
        sub = {}
        for lang in ["en", "zh"]:
            m = (test["language"].values == lang)
            sub[lang] = {}
            for t in CONSISTENCY_TASKS:
                sub[lang][t] = macro_f1(idx_to_labels(t, y[t][m]), idx_to_labels(t, pr[t][m]),
                                        LABEL_SPACES[t])
            sub[lang]["composite"] = float(np.mean([sub[lang][t] for t in CONSISTENCY_TASKS]))
            sub[lang]["n"] = int(m.sum())
            sub[lang]["mean_consistency"] = float(np.mean([(po[t][m] == pr[t][m]).mean()
                                                           for t in CONSISTENCY_TASKS]))
        # source_family is the same partition as language (verified in data audit)
        sf = {}
        for s in sorted(test["source_family"].unique()):
            m = (test["source_family"].values == s)
            vals = [macro_f1(idx_to_labels(t, y[t][m]), idx_to_labels(t, pr[t][m]), LABEL_SPACES[t])
                    for t in CONSISTENCY_TASKS]
            sf[s] = {"composite": float(np.mean(vals)), "n": int(m.sum())}

        # medication rare class (service_domain)
        med_support = int((test["service_domain"].values == "medication").sum())
        med_row = [r for r in per_class_metrics(idx_to_labels("service_domain", y["service_domain"]),
                                               idx_to_labels("service_domain", pr["service_domain"]),
                                               LABEL_SPACES["service_domain"])
                   if r["class"] == "medication"][0]

        metric_rows.append({
            "run_label": RUN_LABEL, "selected_lambda": lam, "seed": seed,
            "test_rows": len(test),
            "composite_macro_f1": comp,
            "service_domain_macro_f1": f1_red["service_domain"],
            "sentiment_macro_f1": f1_red["sentiment"],
            "priority_macro_f1": f1_red["priority"],
            "pii_risk_macro_f1": f1_red["pii_risk"],
            "composite_macro_f1_original_view": comp_orig,
            "mean_consistency_all": mean_cons,
            "mean_consistency_changed": mean_cons_changed,
            "mean_consistency_unchanged": mean_cons_unchanged,
            "changed_records": int(changed.sum()),
            "unchanged_records": int((~changed).sum()),
            "en_composite_macro_f1": sub["en"]["composite"],
            "zh_composite_macro_f1": sub["zh"]["composite"],
            "en_mean_consistency": sub["en"]["mean_consistency"],
            "zh_mean_consistency": sub["zh"]["mean_consistency"],
            "english_public_hospital_reviews_composite": sf["english_public_hospital_reviews"]["composite"],
            "chinese_public_web_deidentified_composite": sf["chinese_public_web_deidentified"]["composite"],
            "medication_support": med_support,
            "medication_f1": med_row["f1"],
            "medication_precision": med_row["precision"],
            "medication_recall": med_row["recall"],
        })

        # per-record predictions (both views)
        pf = pd.DataFrame({
            "record_id": test["record_id"], "seed": seed,
            "language": test["language"], "source_family": test["source_family"],
            "group_id": test["group_id"], "changed_text": changed,
        })
        for t in LABEL_SPACES:
            if t == "pii_risk":
                pf["pii_original_label"] = test["pii_original"]
                pf["pii_redacted_label"] = test["pii_redacted"]
                pf["pred_pii_original_view"] = idx_to_labels(t, po[t])
                pf["pred_pii_redacted_view"] = idx_to_labels(t, pr[t])
            else:
                pf[f"{t}_label"] = test[t]
                pf[f"pred_{t}_original_view"] = idx_to_labels(t, po[t])
                pf[f"pred_{t}_redacted_view"] = idx_to_labels(t, pr[t])
        pred_frames.append(pf)

        # per-class metrics (redacted view)
        for t in LABEL_SPACES:
            tgt = y["pii_risk_redacted"] if t == "pii_risk" else y[t]
            for r in per_class_metrics(idx_to_labels(t, tgt), idx_to_labels(t, pr[t]),
                                       LABEL_SPACES[t]):
                class_rows.append({"run_label": RUN_LABEL, "selected_lambda": lam,
                                   "seed": seed, "task": t, **r})

        # paired bootstrap 95% CI (same resample for all metrics per replicate)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(test)
        boot = {m: [] for m in ["composite_macro_f1", "service_domain_macro_f1",
                                "sentiment_macro_f1", "priority_macro_f1",
                                "mean_consistency_all"]}
        ystr = {t: idx_to_labels(t, y[t]) for t in CONSISTENCY_TASKS}
        pstr = {t: idx_to_labels(t, pr[t]) for t in CONSISTENCY_TASKS}
        for _ in range(BOOTSTRAP_REPLICATES):
            idx = rng.integers(0, n, size=n)
            f1s = {t: macro_f1(ystr[t][idx], pstr[t][idx], LABEL_SPACES[t])
                   for t in CONSISTENCY_TASKS}
            boot["composite_macro_f1"].append(float(np.mean(list(f1s.values()))))
            for t in CONSISTENCY_TASKS:
                boot[f"{t}_macro_f1"].append(f1s[t])
            boot["mean_consistency_all"].append(float(np.mean(
                [(po[t][idx] == pr[t][idx]).mean() for t in CONSISTENCY_TASKS])))
        for m, vals in boot.items():
            lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            boot_rows.append({"run_label": RUN_LABEL, "selected_lambda": lam, "seed": seed,
                              "metric": m, "ci_lower": float(lo), "ci_upper": float(hi),
                              "replicates": BOOTSTRAP_REPLICATES,
                              "method": "paired bootstrap over locked-test records (percentile 95%)"})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(MODEL_DIR, "locked_test_metrics_by_seed.csv"), index=False)
    pd.concat(pred_frames).to_csv(os.path.join(MODEL_DIR, "locked_test_predictions.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(os.path.join(MODEL_DIR, "per_class_metrics.csv"), index=False)
    pd.DataFrame(boot_rows).to_csv(os.path.join(MODEL_DIR, "bootstrap_confidence_intervals.csv"), index=False)

    # mean / std across seeds
    num_cols = [c for c in metrics_df.columns if c not in
                {"run_label", "selected_lambda", "seed"}]
    summary = metrics_df.groupby("selected_lambda")[num_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "run_label", RUN_LABEL)
    summary.to_csv(os.path.join(MODEL_DIR, "locked_test_summary_across_seeds.csv"), index=False)

    print(metrics_df[["seed", "composite_macro_f1", "service_domain_macro_f1",
                      "sentiment_macro_f1", "priority_macro_f1", "pii_risk_macro_f1",
                      "mean_consistency_all", "mean_consistency_changed",
                      "mean_consistency_unchanged", "medication_support", "medication_f1"]].to_string(index=False))
    print(summary.to_string(index=False))
    print("stage3 done")


if __name__ == "__main__":
    main()
