"""Stage 2: train the corrected TF-IDF student over the lambda grid x seeds on
the locked train split (160 FULL-BATCH epochs, Adam lr=0.006, weight decay
1e-4); evaluate on the locked VALIDATION split only; then apply the
preregistered selection rule to lock lambda.

Storage rules: NO checkpoints, NO tfidf_vectorizer.pkl are written. The
vectorizer is refit deterministically from the locked train split in every
stage. Results are appended to model/validation_lambda_results.csv after each
run, so the stage is resumable (skips completed (lambda, seed) rows).

The selection rule is preregistered in model/selection_config.yaml BEFORE any
training starts. Locked-test labels/predictions are never touched here.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, LOGS_DIR, RUN_ROOT, LABEL_SPACES,
                    ALL_TASKS, CONSISTENCY_TASKS, LAMBDA_GRID, SEEDS, EPOCHS,
                    FULL_BATCH, CHAR_FEATURES, WORD_FEATURES, SHARED_DIM,
                    DROPOUT, LR, WEIGHT_DECAY, SELECTION_TOLERANCE, LOG_EVERY,
                    RUN_LABEL, RUN_DATE, write_json, ensure_dirs)
from metrics_utils import macro_f1, composite_macro_f1, idx_to_labels
from student import TfidfStudent

RESULTS_CSV = MODEL_DIR / "validation_lambda_results.csv"
LOG_DIR = LOGS_DIR / "stage2_training"


# ---------------------------------------------------------------------------
# features / targets (identical deterministic construction in every stage)
# ---------------------------------------------------------------------------
def build_features(df):
    """Fit char_wb (2,5) TF-IDF (max 6000) + word (1,2) TF-IDF (max 4000) on the
    ORIGINAL and REDACTED TRAIN views only; transform all splits/views."""
    train = df[df["locked_split"] == "train"]
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           max_features=CHAR_FEATURES, min_df=2, dtype=np.float32)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=WORD_FEATURES, min_df=2, lowercase=True,
                           dtype=np.float32)
    fit_text = pd.concat([train["original_text"], train["redacted_text"]]).astype(str)
    char.fit(fit_text)
    word.fit(fit_text)

    def transform(texts):
        return hstack([char.transform(texts.astype(str)),
                       word.transform(texts.astype(str))]).astype(np.float32).tocsr()

    feats = {}
    for split in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == split].reset_index(drop=True)
        feats[split] = {"Xo": transform(sub["original_text"]),
                        "Xr": transform(sub["redacted_text"]), "n": len(sub)}
    feature_dim = len(char.vocabulary_) + len(word.vocabulary_)
    return feats, feature_dim, len(char.vocabulary_), len(word.vocabulary_)


def build_targets(df):
    targets = {}
    for split in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == split].reset_index(drop=True)
        t = {}
        for task in LABEL_SPACES:
            if task == "pii_risk":
                t["pii_risk_original"] = sub["pii_original"].map(
                    LABEL_SPACES["pii_risk"].index).values.astype(np.int64)
                t["pii_risk_redacted"] = sub["pii_redacted"].map(
                    LABEL_SPACES["pii_risk"].index).values.astype(np.int64)
            else:
                t[task] = sub[task].map(
                    LABEL_SPACES[task].index).values.astype(np.int64)
        targets[split] = t
    return targets


def evaluate(model, feats_split, targets_split):
    """Validation-style evaluation on the REDACTED view (inference convention)."""
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
    cons = float(np.mean([(po[t] == pr[t]).mean() for t in CONSISTENCY_TASKS]))
    return comp, f1s, cons


def train_one(lam, seed, feats, targets, epochs, log_path):
    """160 full-batch epochs (one update per epoch over the whole train split)."""
    tr = feats["train"]
    n = tr["n"]
    model = TfidfStudent(feats["feature_dim"], seed)
    log_lines = [f"run_label={RUN_LABEL} lambda={lam} seed={seed} epochs={epochs} "
                 f"full_batch={FULL_BATCH} lr={LR} weight_decay={WEIGHT_DECAY} "
                 f"dropout={DROPOUT} shared_dim={SHARED_DIM}"]
    t0 = time.time()
    for epoch in range(epochs):
        Xo, Xr = tr["Xo"], tr["Xr"]
        y_o = {t: targets["train"][t] for t in ALL_TASKS if t != "pii_risk"}
        y_o["pii_risk"] = targets["train"]["pii_risk_original"]
        y_r = {t: targets["train"][t] for t in ALL_TASKS if t != "pii_risk"}
        y_r["pii_risk"] = targets["train"]["pii_risk_redacted"]
        loss, kl, grads = model.loss_and_grads(Xo, Xr, y_o, y_r, lam)
        model.step(grads)
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            comp, f1s, cons = evaluate(model, feats["validation"], targets["validation"])
            log_lines.append(
                f"epoch={epoch+1:3d} train_loss={loss:.4f} consistency_kl={kl:.4f} "
                f"val_composite_macro_f1={comp:.4f} val_mean_consistency={cons:.4f}")
    comp, f1s, cons = evaluate(model, feats["validation"], targets["validation"])
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


# ---------------------------------------------------------------------------
# preregistered selection config (written BEFORE training)
# ---------------------------------------------------------------------------
def write_selection_config():
    cfg = f"""# Preregistered model-selection configuration (written before any training).
run_label: {RUN_LABEL}
date: {RUN_DATE}
selection_data: locked VALIDATION split only (locked-test labels/predictions are never used)
lambda_grid: {LAMBDA_GRID}
seeds: {SEEDS}
rule: >-
  maximize mean validation composite macro-F1 (mean over seeds);
  within tolerance {SELECTION_TOLERANCE} of the best, maximize mean validation
  consistency; then choose the smaller lambda.
tolerance: {SELECTION_TOLERANCE}
tie_break: smaller lambda
training:
  epochs: {EPOCHS}            # manuscript: 160 full-batch epochs
  batching: full-batch        # one update per epoch over the whole train split
  optimizer: Adam
  lr: {LR}
  weight_decay: {WEIGHT_DECAY}
  dropout: {DROPOUT}
  shared_dim: {SHARED_DIM}
"""
    path = MODEL_DIR / "selection_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg)
    return path


def write_implementation_assumptions():
    text = """# Implementation Assumptions — CORRECTED_FAITHFUL_REIMPLEMENTATION

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
"""
    path = RUN_ROOT / "implementation_assumptions.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDA_GRID))
    ap.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    ensure_dirs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = write_selection_config()
    assumptions_path = write_implementation_assumptions()
    print(f"preregistered selection config: {cfg_path}")
    print(f"implementation assumptions: {assumptions_path}")

    df = pd.read_csv(DATA_DIR / "locked_split.csv")
    feats, feature_dim, char_n, word_n = build_features(df)
    feats["feature_dim"] = feature_dim
    targets = build_targets(df)
    print(f"feature_dim={feature_dim} (char={char_n} word={word_n}); "
          f"train={feats['train']['n']} val={feats['validation']['n']} test={feats['test']['n']}")
    print(f"training: {args.epochs} full-batch epochs, Adam lr={LR}, wd={WEIGHT_DECAY}")

    rows = []
    if RESULTS_CSV.exists():
        rows = pd.read_csv(RESULTS_CSV).to_dict("records")
    done = {(r["lambda"], r["seed"]) for r in rows}

    for lam in lambdas:
        for seed in seeds:
            if (lam, seed) in done:
                print(f"skip existing lambda={lam} seed={seed}")
                continue
            log_path = LOG_DIR / f"run_lambda_{lam}_seed_{seed}.log"
            model, comp, f1s, cons = train_one(lam, seed, feats, targets,
                                               args.epochs, log_path)
            del model  # release immediately (storage rule: fit and release)
            rows.append({"run_label": RUN_LABEL, "lambda": lam, "seed": seed,
                         "epochs": args.epochs, "batching": "full-batch",
                         "feature_dim": feature_dim,
                         "val_composite_macro_f1": comp,
                         "val_service_domain_macro_f1": f1s["service_domain"],
                         "val_sentiment_macro_f1": f1s["sentiment"],
                         "val_priority_macro_f1": f1s["priority"],
                         "val_pii_risk_macro_f1": f1s["pii_risk"],
                         "val_mean_consistency": cons})
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
            print(f"done lambda={lam} seed={seed} val_composite={comp:.4f} "
                  f"val_consistency={cons:.4f}")

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
        "rule": "max validation composite macro-F1; within tolerance 0.005 max "
                "validation mean consistency; then smaller lambda",
        "tolerance": SELECTION_TOLERANCE,
        "per_lambda_validation": agg.to_dict("records"),
        "best_composite_macro_f1": float(best_comp),
        "selected_lambda": selected,
        "selection_data": "locked validation split only; locked-test labels/predictions not used",
        "selected_at": RUN_DATE,
        "training": {"epochs": args.epochs, "batching": "full-batch", "lr": LR,
                     "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
                     "shared_dim": SHARED_DIM, "seeds": seeds},
    }
    write_json(MODEL_DIR / "selection_log.json", selection)
    with open(MODEL_DIR / "selection_log.md", "w", encoding="utf-8") as f:
        f.write("# Lambda Selection Log (validation only)\n\n")
        f.write("Rule (preregistered in `model/selection_config.yaml` before training): "
                "maximize validation composite macro-F1; within tolerance 0.005 maximize "
                "validation mean consistency; then choose smaller lambda. "
                "Locked-test labels/predictions were never used.\n\n")
        f.write(agg.to_string(index=False))
        f.write(f"\n\nBest validation composite macro-F1: {best_comp:.6f}\n")
        f.write(f"**Selected lambda: {selected}**\n")
    print("SELECTED LAMBDA:", selected)
    print(agg.to_string(index=False))
    print("stage2 done")


if __name__ == "__main__":
    main()
