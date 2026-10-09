"""Stage 2: train the TF-IDF student over the lambda grid x seeds on the locked
train split; evaluate on the locked validation split only; save checkpoints,
per-run logs, and validation_lambda_results.csv; then apply the preregistered
selection rule (model/selection_config.yaml) to lock lambda.

Resume-safe: skips (lambda, seed) combos whose checkpoint already exists.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, LABEL_SPACES, CONSISTENCY_TASKS,
                    LAMBDA_GRID, SEEDS, EPOCHS, BATCH_SIZE, CHAR_FEATURES,
                    WORD_FEATURES, SELECTION_TOLERANCE, RUN_LABEL,
                    write_json)
from metrics_utils import macro_f1, composite_macro_f1, idx_to_labels
from student import TfidfStudent

CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
LOG_DIR = os.path.join(MODEL_DIR, "training_logs")
RESULTS_CSV = os.path.join(MODEL_DIR, "validation_lambda_results.csv")
VEC_PATH = os.path.join(CKPT_DIR, "tfidf_vectorizer.pkl")


def build_features(df):
    train = df[df["locked_split"] == "train"]
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           max_features=CHAR_FEATURES, min_df=2, dtype=np.float32)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=WORD_FEATURES, min_df=2, lowercase=True,
                           dtype=np.float32)
    fit_text = pd.concat([train["original_text"], train["redacted_text"]]).astype(str)
    char.fit(fit_text)
    word.fit(fit_text)
    with open(VEC_PATH, "wb") as f:
        pickle.dump({"char": char, "word": word}, f)

    def transform(texts):
        return hstack([char.transform(texts.astype(str)),
                       word.transform(texts.astype(str))]).astype(np.float32).tocsr()

    feats = {}
    for split in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == split]
        feats[split] = {
            "Xo": transform(sub["original_text"]), "Xr": transform(sub["redacted_text"]),
            "idx": np.arange(len(sub)), "n": len(sub),
        }
    feature_dim = len(char.vocabulary_) + len(word.vocabulary_)
    return feats, feature_dim


def build_targets(df):
    targets = {}
    for split in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == split]
        t = {}
        for task in LABEL_SPACES:
            if task == "pii_risk":
                t["pii_risk_original"] = sub["pii_original"].map(
                    lambda x: LABEL_SPACES["pii_risk"].index(x)).values.astype(np.int64)
                t["pii_risk_redacted"] = sub["pii_redacted"].map(
                    lambda x: LABEL_SPACES["pii_risk"].index(x)).values.astype(np.int64)
            else:
                t[task] = sub[task].map(
                    lambda x: LABEL_SPACES[task].index(x)).values.astype(np.int64)
        targets[split] = t
    return targets


def evaluate(model, feats_split, targets_split):
    po = model.predict_idx(feats_split["Xo"])
    pr = model.predict_idx(feats_split["Xr"])
    f1s = {}
    for task in LABEL_SPACES:
        if task == "pii_risk":
            y = targets_split["pii_risk_redacted"]
        else:
            y = targets_split[task]
        f1s[task] = macro_f1(idx_to_labels(task, y), idx_to_labels(task, pr[task]),
                             LABEL_SPACES[task])
    comp = composite_macro_f1(f1s)
    cons = np.mean([(po[t] == pr[t]).mean() for t in CONSISTENCY_TASKS])
    return comp, f1s, float(cons), po, pr


def train_one(lam, seed, feats, targets, epochs, log_path):
    tr = feats["train"]
    n = tr["n"]
    model = TfidfStudent(feats["feature_dim"], seed)
    rng = np.random.default_rng(seed)
    log_lines = [f"run_label={RUN_LABEL} lambda={lam} seed={seed} epochs={epochs} "
                 f"batch={BATCH_SIZE} lr=0.006 wd=1e-4 dropout=0.15 shared=384"]
    t0 = time.time()
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss, total_kl, nb = 0.0, 0.0, 0
        for s in range(0, n, BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            Xo = tr["Xo"][idx]
            Xr = tr["Xr"][idx]
            y_o = {t: targets["train"][t][idx] for t in LABEL_SPACES if t != "pii_risk"}
            y_o["pii_risk"] = targets["train"]["pii_risk_original"][idx]
            y_r = {t: targets["train"][t][idx] for t in LABEL_SPACES if t != "pii_risk"}
            y_r["pii_risk"] = targets["train"]["pii_risk_redacted"][idx]
            loss, kl, grads = model.loss_and_grads(Xo, Xr, y_o, y_r, lam)
            model.step(grads)
            total_loss += loss
            total_kl += kl
            nb += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            comp, f1s, cons, _, _ = evaluate(model, feats["validation"], targets["validation"])
            log_lines.append(
                f"epoch={epoch+1:3d} train_loss={total_loss/nb:.4f} "
                f"consistency_kl={total_kl/nb:.4f} val_composite_macro_f1={comp:.4f} "
                f"val_mean_consistency={cons:.4f}")
    comp, f1s, cons, _, _ = evaluate(model, feats["validation"], targets["validation"])
    log_lines.append(f"final val_composite_macro_f1={comp:.6f} "
                     f"val_service_domain_macro_f1={f1s['service_domain']:.6f} "
                     f"val_sentiment_macro_f1={f1s['sentiment']:.6f} "
                     f"val_priority_macro_f1={f1s['priority']:.6f} "
                     f"val_pii_risk_macro_f1={f1s['pii_risk']:.6f} "
                     f"val_mean_consistency={cons:.6f} "
                     f"train_seconds={time.time()-t0:.1f}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    return model, comp, f1s, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDA_GRID))
    ap.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, "locked_split.csv"))
    if os.path.exists(VEC_PATH):
        with open(VEC_PATH, "rb") as f:
            vecs = pickle.load(f)
        char, word = vecs["char"], vecs["word"]
        def transform(texts):
            return hstack([char.transform(texts.astype(str)),
                           word.transform(texts.astype(str))]).astype(np.float32).tocsr()
        feats = {}
        for split in ["train", "validation", "test"]:
            sub = df[df["locked_split"] == split]
            feats[split] = {"Xo": transform(sub["original_text"]),
                            "Xr": transform(sub["redacted_text"]), "n": len(sub)}
        feature_dim = len(char.vocabulary_) + len(word.vocabulary_)
        char_n, word_n = len(char.vocabulary_), len(word.vocabulary_)
    else:
        feats, feature_dim, char_n, word_n = build_features(df)
    feats["feature_dim"] = feature_dim
    targets = build_targets(df)
    print(f"feature_dim={feature_dim} (char={char_n} word={word_n})")

    rows = []
    if os.path.exists(RESULTS_CSV):
        rows = pd.read_csv(RESULTS_CSV).to_dict("records")
    done = {(r["lambda"], r["seed"]) for r in rows}

    for lam in lambdas:
        for seed in seeds:
            if (lam, seed) in done:
                print(f"skip existing lambda={lam} seed={seed}")
                continue
            ckpt = os.path.join(CKPT_DIR, f"student_lambda_{lam}_seed_{seed}.npz")
            log_path = os.path.join(LOG_DIR, f"run_lambda_{lam}_seed_{seed}.log")
            model, comp, f1s, cons = train_one(lam, seed, feats, targets, args.epochs, log_path)
            model.save(ckpt)
            rows.append({"run_label": RUN_LABEL, "lambda": lam, "seed": seed,
                         "epochs": args.epochs, "batch_size": BATCH_SIZE,
                         "feature_dim": feature_dim,
                         "val_composite_macro_f1": comp,
                         "val_service_domain_macro_f1": f1s["service_domain"],
                         "val_sentiment_macro_f1": f1s["sentiment"],
                         "val_priority_macro_f1": f1s["priority"],
                         "val_pii_risk_macro_f1": f1s["pii_risk"],
                         "val_mean_consistency": cons})
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
            print(f"done lambda={lam} seed={seed} val_composite={comp:.4f} "
                  f"val_consistency={cons:.4f} -> {ckpt}")

    # ---------------- preregistered selection (validation only) ----------------
    res = pd.DataFrame(rows)
    agg = res.groupby("lambda").agg(
        val_composite_macro_f1_mean=("val_composite_macro_f1", "mean"),
        val_composite_macro_f1_std=("val_composite_macro_f1", "std"),
        val_mean_consistency_mean=("val_mean_consistency", "mean"),
        val_mean_consistency_std=("val_mean_consistency", "std"),
        seeds=("seed", "count")).reset_index()
    best_comp = agg["val_composite_macro_f1_mean"].max()
    within = agg[agg["val_composite_macro_f1_mean"] >= best_comp - SELECTION_TOLERANCE]
    best_cons = within["val_mean_consistency_mean"].max()
    tied = within[within["val_mean_consistency_mean"] >= best_cons - 1e-12]
    selected = float(tied["lambda"].min())
    selection = {
        "run_label": RUN_LABEL,
        "rule": "max validation composite macro-F1; within tolerance 0.005 max validation mean consistency; then smaller lambda",
        "tolerance": SELECTION_TOLERANCE,
        "per_lambda_validation": agg.to_dict("records"),
        "best_composite_macro_f1": float(best_comp),
        "selected_lambda": selected,
        "selection_data": "locked validation split only; locked-test labels/predictions not used",
        "selected_at": "2026-10-09",
    }
    write_json(os.path.join(LOG_DIR, "selection_log.json"), selection)
    with open(os.path.join(LOG_DIR, "selection_log.md"), "w", encoding="utf-8") as f:
        f.write("# Lambda Selection Log (validation only)\n\n")
        f.write("Rule (preregistered in `model/selection_config.yaml` before test predictions): "
                "maximize validation composite macro-F1; within tolerance 0.005 maximize "
                "validation mean consistency; then choose smaller lambda.\n\n")
        f.write(agg.to_string(index=False))
        f.write(f"\n\nBest validation composite macro-F1: {best_comp:.6f}\n")
        f.write(f"**Selected lambda: {selected}**\n")
    print("SELECTED LAMBDA:", selected)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
