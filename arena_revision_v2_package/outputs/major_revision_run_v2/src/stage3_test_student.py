"""Stage 3: locked-test evaluation of the selected lambda.

After validation-only lambda selection, each selected seed is RETRAINED in
memory (160 full-batch epochs; no checkpoints are loaded or saved) and the
locked test is evaluated EXACTLY ONCE per seed.

Produces:
- model/locked_test_metrics_by_seed.csv     (per-seed task/composite/consistency/subgroup metrics)
- model/locked_test_summary_across_seeds.csv (mean/std across seeds)
- model/locked_test_predictions.csv         (ONE combined RC per-record prediction CSV, all seeds)
- model/per_class_metrics.csv               (per-class precision/recall/F1/support per seed)
- model/bootstrap_confidence_intervals.csv  (paired bootstrap 95% CIs, 2000 replicates)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, LABEL_SPACES, ALL_TASKS,
                    CONSISTENCY_TASKS, SEEDS, EPOCHS, CHAR_FEATURES,
                    WORD_FEATURES, SHARED_DIM, DROPOUT, LR, WEIGHT_DECAY,
                    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, RUN_LABEL, RUN_DATE,
                    write_json, read_json, ensure_dirs)
from metrics_utils import macro_f1, per_class_metrics, idx_to_labels
from stage2_train_student import build_features, build_targets, train_one, LOG_DIR


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_DIR / "locked_split.csv")
    test = df[df["locked_split"] == "test"].reset_index(drop=True)

    selection = read_json(MODEL_DIR / "selection_log.json")
    lam = float(selection["selected_lambda"])
    seeds = [int(s) for s in selection.get("seeds", SEEDS)]
    print(f"selected lambda={lam}; retraining seeds {seeds} in memory and "
          f"evaluating the locked test exactly once per seed")

    feats, feature_dim, char_n, word_n = build_features(df)
    feats["feature_dim"] = feature_dim
    targets = build_targets(df)
    print(f"feature_dim={feature_dim} (char={char_n} word={word_n}); test rows={len(test)}")

    changed = (test["original_text"].astype(str) !=
               test["redacted_text"].astype(str)).values

    y = {}
    for task in LABEL_SPACES:
        if task == "pii_risk":
            y["pii_risk_original"] = test["pii_original"].map(LABEL_SPACES[task].index).values
            y["pii_risk_redacted"] = test["pii_redacted"].map(LABEL_SPACES[task].index).values
        else:
            y[task] = test[task].map(LABEL_SPACES[task].index).values

    metric_rows, pred_frames, class_rows, boot_rows = [], [], [], []
    for seed in seeds:
        t0 = time.time()
        # retrain in memory (no checkpoint is loaded or saved)
        log_path = LOG_DIR / f"test_retrain_lambda_{lam}_seed_{seed}.log"
        model, _, _, _ = train_one(lam, seed, feats, targets, EPOCHS, log_path)
        print(f"seed {seed} retrained in {time.time()-t0:.1f}s; evaluating locked test once")

        Xo, Xr = feats["test"]["Xo"], feats["test"]["Xr"]
        po = model.predict_idx(Xo)
        pr = model.predict_idx(Xr)
        del model  # release immediately

        # primary metrics on the redacted view (inference convention)
        f1_red = {}
        for task in LABEL_SPACES:
            tgt = y["pii_risk_redacted"] if task == "pii_risk" else y[task]
            f1_red[task] = macro_f1(idx_to_labels(task, tgt),
                                    idx_to_labels(task, pr[task]), LABEL_SPACES[task])
        comp = float(np.mean([f1_red[t] for t in CONSISTENCY_TASKS]))
        f1_orig = {}
        for task in LABEL_SPACES:
            tgt = y["pii_risk_original"] if task == "pii_risk" else y[task]
            f1_orig[task] = macro_f1(idx_to_labels(task, tgt),
                                     idx_to_labels(task, po[task]), LABEL_SPACES[task])
        comp_orig = float(np.mean([f1_orig[t] for t in CONSISTENCY_TASKS]))

        # consistency: all / changed / unchanged
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
                sub[lang][t] = macro_f1(idx_to_labels(t, y[t][m]),
                                        idx_to_labels(t, pr[t][m]), LABEL_SPACES[t])
            sub[lang]["composite"] = float(np.mean([sub[lang][t] for t in CONSISTENCY_TASKS]))
            sub[lang]["n"] = int(m.sum())
            sub[lang]["mean_consistency"] = float(np.mean(
                [(po[t][m] == pr[t][m]).mean() for t in CONSISTENCY_TASKS]))
            sub[lang]["pii_risk_macro_f1"] = macro_f1(
                idx_to_labels("pii_risk", y["pii_risk_redacted"][m]),
                idx_to_labels("pii_risk", pr["pii_risk"][m]), LABEL_SPACES["pii_risk"])
        sf = {}
        for s in sorted(test["source_family"].unique()):
            m = (test["source_family"].values == s)
            vals = [macro_f1(idx_to_labels(t, y[t][m]), idx_to_labels(t, pr[t][m]),
                             LABEL_SPACES[t]) for t in CONSISTENCY_TASKS]
            sf[s] = {"composite": float(np.mean(vals)), "n": int(m.sum())}

        # rare class: medication (service_domain)
        med_support = int((test["service_domain"].values == "medication").sum())
        med_row = [r for r in per_class_metrics(
            idx_to_labels("service_domain", y["service_domain"]),
            idx_to_labels("service_domain", pr["service_domain"]),
            LABEL_SPACES["service_domain"]) if r["class"] == "medication"][0]

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
            "en_mean_consistency": sub["en"]["mean_consistency"],
            "en_pii_risk_macro_f1": sub["en"]["pii_risk_macro_f1"],
            "zh_composite_macro_f1": sub["zh"]["composite"],
            "zh_mean_consistency": sub["zh"]["mean_consistency"],
            "zh_pii_risk_macro_f1": sub["zh"]["pii_risk_macro_f1"],
            "en_n": sub["en"]["n"], "zh_n": sub["zh"]["n"],
            "english_public_hospital_reviews_composite": sf["english_public_hospital_reviews"]["composite"],
            "chinese_public_web_deidentified_composite": sf["chinese_public_web_deidentified"]["composite"],
            "medication_support": med_support,
            "medication_f1": med_row["f1"],
            "medication_precision": med_row["precision"],
            "medication_recall": med_row["recall"],
        })

        # ONE combined per-record prediction CSV (all seeds)
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
            for r in per_class_metrics(idx_to_labels(t, tgt),
                                       idx_to_labels(t, pr[t]), LABEL_SPACES[t]):
                class_rows.append({"run_label": RUN_LABEL, "selected_lambda": lam,
                                   "seed": seed, "task": t, **r})

        # paired bootstrap 95% CI (same resample for all metrics per replicate)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(test)
        boot = {m: [] for m in ["composite_macro_f1", "service_domain_macro_f1",
                                "sentiment_macro_f1", "priority_macro_f1",
                                "pii_risk_macro_f1", "mean_consistency_all",
                                "mean_consistency_changed"]}
        ystr = {t: idx_to_labels(t, y[t]) for t in CONSISTENCY_TASKS}
        pstr = {t: idx_to_labels(t, pr[t]) for t in CONSISTENCY_TASKS}
        ystr_pii = idx_to_labels("pii_risk", y["pii_risk_redacted"])
        pstr_pii = idx_to_labels("pii_risk", pr["pii_risk"])
        for _ in range(BOOTSTRAP_REPLICATES):
            idx = rng.integers(0, n, size=n)
            f1s = {t: macro_f1(ystr[t][idx], pstr[t][idx], LABEL_SPACES[t])
                   for t in CONSISTENCY_TASKS}
            boot["composite_macro_f1"].append(float(np.mean(list(f1s.values()))))
            for t in CONSISTENCY_TASKS:
                boot[f"{t}_macro_f1"].append(f1s[t])
            boot["pii_risk_macro_f1"].append(
                macro_f1(ystr_pii[idx], pstr_pii[idx], LABEL_SPACES["pii_risk"]))
            boot["mean_consistency_all"].append(float(np.mean(
                [(po[t][idx] == pr[t][idx]).mean() for t in CONSISTENCY_TASKS])))
            boot["mean_consistency_changed"].append(float(np.mean(
                [(po[t][idx] == pr[t][idx])[changed[idx]].mean()
                 for t in CONSISTENCY_TASKS])))
        for m, vals in boot.items():
            lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            boot_rows.append({"run_label": RUN_LABEL, "selected_lambda": lam,
                              "seed": seed, "metric": m, "ci_lower": float(lo),
                              "ci_upper": float(hi),
                              "replicates": BOOTSTRAP_REPLICATES,
                              "method": "paired bootstrap over locked-test records (percentile 95%)"})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(MODEL_DIR / "locked_test_metrics_by_seed.csv", index=False)
    pd.concat(pred_frames).to_csv(MODEL_DIR / "locked_test_predictions.csv", index=False)
    pd.DataFrame(class_rows).to_csv(MODEL_DIR / "per_class_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(MODEL_DIR / "bootstrap_confidence_intervals.csv",
                                   index=False)

    num_cols = [c for c in metrics_df.columns
                if c not in {"run_label", "selected_lambda", "seed"}]
    summary = metrics_df.groupby("selected_lambda")[num_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "run_label", RUN_LABEL)
    summary.to_csv(MODEL_DIR / "locked_test_summary_across_seeds.csv", index=False)

    print(metrics_df[["seed", "composite_macro_f1", "service_domain_macro_f1",
                      "sentiment_macro_f1", "priority_macro_f1", "pii_risk_macro_f1",
                      "mean_consistency_all", "mean_consistency_changed",
                      "mean_consistency_unchanged", "medication_support",
                      "medication_f1"]].to_string(index=False))
    print(summary.to_string(index=False))
    print("stage3 done")


if __name__ == "__main__":
    main()
