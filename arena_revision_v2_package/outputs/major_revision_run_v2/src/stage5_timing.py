"""Stage 5: common end-to-end CPU timing for RC-EdgeDistill and all runnable
TF-IDF baselines.

Identical boundary for every method, per record: input normalization (NFKC +
whitespace collapse) -> local redaction (regex rules) -> TF-IDF construction
(fitted vectorizer transform) -> inference -> decoding (argmax/label -> string)
-> output serialization (JSON). Model-only latency is NOT mixed in.

Protocol: CPU only, OMP_NUM_THREADS=1, same record order (locked-test file
order), per-record (batch=1), 1 warm-up pass + 5 measured repetitions, timer
time.perf_counter. Timing predictions are never used for any metric.

Storage rules: only the AGGREGATE timing summary and the protocol manifest are
saved (no raw timing files, no per-record latency files). The student and the
baselines are refit in memory (selected lambda / selected hyperparameters,
seed 20260715) and released after timing.
"""
import json
import os
import platform
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, BASELINES_DIR, TIMING_DIR, LABEL_SPACES,
                    ALL_TASKS, CONSISTENCY_TASKS, CHAR_FEATURES, WORD_FEATURES,
                    RUN_LABEL, RUN_DATE, redact_text, sha256_file, read_json,
                    ensure_dirs)
from stage2_train_student import build_features, build_targets, train_one
from stage4_baselines import GRIDS
from student import TfidfStudent

WARMUP = 1
REPS = 5
TIMING_SEED = 20260715  # student seed and baseline fits used for timing


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_DIR / "locked_split.csv")
    test = df[df["locked_split"] == "test"].reset_index(drop=True)
    raw_texts = test["original_text"].astype(str).tolist()

    feats, feature_dim, char_n, word_n = build_features(df)
    feats["feature_dim"] = feature_dim
    targets = build_targets(df)

    selection = read_json(MODEL_DIR / "selection_log.json")
    lam = float(selection["selected_lambda"])

    # ---- refit the student in memory (selected lambda, timing seed) ----
    from common import LOGS_DIR
    log_path = LOGS_DIR / "stage2_training" / f"timing_retrain_lambda_{lam}_seed_{TIMING_SEED}.log"
    student, _, _, _ = train_one(lam, TIMING_SEED, feats, targets, 160, log_path)
    print(f"student refit in memory (lambda={lam}, seed={TIMING_SEED})")

    # ---- refit baselines in memory (selected hyperparameters, timing seed) ----
    hyper = pd.read_csv(BASELINES_DIR / "hyperparameters.csv")
    baseline_models = {}
    Xr_train = feats["train"]["Xr"]
    for name, spec in GRIDS.items():
        sel = hyper[hyper["baseline"] == name].iloc[0]
        val = float(sel["selected_value"])
        models = {}
        for t in CONSISTENCY_TASKS:
            clf = spec["factory"](val)
            if name == "tfidf_sgd_modified_huber":
                clf.set_params(random_state=TIMING_SEED)
            clf.fit(Xr_train, df[df["locked_split"] == "train"][t].values)
            models[t] = clf
        baseline_models[name] = models
        print(f"baseline refit in memory: {name} ({sel['parameter']}={val})")

    # vectorizers for the per-record TF-IDF construction (in memory only)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    train = df[df["locked_split"] == "train"]
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           max_features=CHAR_FEATURES, min_df=2, dtype=np.float32)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=WORD_FEATURES, min_df=2, lowercase=True,
                           dtype=np.float32)
    fit_text = pd.concat([train["original_text"], train["redacted_text"]]).astype(str)
    char.fit(fit_text)
    word.fit(fit_text)

    def tfidf(text):
        return hstack([char.transform([text]),
                       word.transform([text])]).astype(np.float32).tocsr()

    def norm(text):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

    def student_pipeline(text):
        t = redact_text(norm(text))
        X = tfidf(t)
        logits, _, _, _ = student.forward(X, train=False)
        out = {task: LABEL_SPACES[task][int(logits[task].argmax(axis=1)[0])]
               for task in LABEL_SPACES}
        return json.dumps(out, ensure_ascii=False)

    def make_baseline_pipeline(models):
        def pipeline(text):
            t = redact_text(norm(text))
            X = tfidf(t)
            out = {task: str(models[task].predict(X)[0]) for task in CONSISTENCY_TASKS}
            return json.dumps(out, ensure_ascii=False)
        return pipeline

    pipelines = {f"rc_edgedistill_lambda_{lam}": student_pipeline}
    for name, models in baseline_models.items():
        pipelines[name] = make_baseline_pipeline(models)
    # NOTE: `student` and `baseline_models` stay alive (the pipeline closures
    # reference them); they are released when this process exits.

    summary_rows = []
    for method, fn in pipelines.items():
        for text in raw_texts:          # warm-up pass
            fn(text)
        rep_means = []
        all_lat = []
        for rep in range(REPS):
            lat = []
            t0 = time.perf_counter()
            for text in raw_texts:
                s = time.perf_counter()
                fn(text)
                lat.append((time.perf_counter() - s) * 1000.0)
            total = time.perf_counter() - t0
            lat = np.array(lat)
            all_lat.extend(lat.tolist())
            rep_means.append(float(lat.mean()))
            print(f"{method} rep={rep} mean={lat.mean():.3f} ms/record total={total:.2f}s")
        all_lat = np.array(all_lat)
        rep_means = np.array(rep_means)
        summary_rows.append({
            "method": method, "records": len(raw_texts),
            "mean_ms_per_record": float(all_lat.mean()),
            "std_ms_per_record": float(all_lat.std()),
            "median_ms_per_record": float(np.median(all_lat)),
            "p95_ms_per_record": float(np.percentile(all_lat, 95)),
            "min_ms_per_record": float(all_lat.min()),
            "max_ms_per_record": float(all_lat.max()),
            "mean_of_repetition_means_ms": float(rep_means.mean()),
            "std_of_repetition_means_ms": float(rep_means.std()),
            "mean_throughput_records_per_second": float(
                (len(raw_texts) * REPS) / (all_lat.sum() / 1000.0)),
            "repetitions": REPS, "warmup_passes": WARMUP,
            "batch_policy": "per-record (batch=1)",
            "threads": 1, "timer": "time.perf_counter",
        })

    pd.DataFrame(summary_rows).to_csv(TIMING_DIR / "end_to_end_timing.csv", index=False)

    import scipy, sklearn
    manifest = {
        "run_label": RUN_LABEL,
        "date": RUN_DATE,
        "scope": "end-to-end per-record serving latency: input normalization -> local "
                 "redaction -> TF-IDF construction -> inference -> decoding -> JSON serialization",
        "not_included": "model fitting/training time; vectorizer fitting; accuracy metrics "
                        "(timing predictions are not used for any metric)",
        "record_order": "locked_split.csv locked-test rows in file order",
        "records": len(raw_texts),
        "batch_policy": "per-record (batch=1)",
        "warmup_passes": WARMUP,
        "repetitions": REPS,
        "timer": "time.perf_counter",
        "threads": "OMP_NUM_THREADS=1 (also pinned during training)",
        "cpu": cpu_model(),
        "cpu_cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "input_normalization": "NFKC + whitespace collapse (identical for all methods)",
        "redaction": "regex rules from common.py (identical to code_reference/redaction.py)",
        "student_config": f"selected lambda {lam}, seed {TIMING_SEED}, refit in memory "
                          "(no checkpoint saved/loaded)",
        "baseline_configs": "selected hyperparameters (validation-only), seed "
                            f"{TIMING_SEED}, refit in memory",
        "data_sha256": sha256_file(DATA_DIR / "locked_split.csv"),
    }
    with open(TIMING_DIR / "timing_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    md = ["# Timing Protocol (common end-to-end boundary)", "",
          f"- Run label: `{RUN_LABEL}` | Date: {RUN_DATE}",
          "- Boundary per record (identical for every method): input normalization "
          "(NFKC + whitespace collapse) -> local redaction (regex rules) -> TF-IDF "
          "construction (fitted vectorizer transform) -> inference -> decoding "
          "(argmax/label -> string) -> output serialization (JSON).",
          "- Model-only latency is NOT mixed in; training/fitting time and vectorizer "
          "fitting are excluded.",
          f"- Records: {len(raw_texts)} locked-test records in file order; per-record "
          f"(batch=1); {WARMUP} warm-up pass + {REPS} measured repetitions; "
          "timer `time.perf_counter`; threads=1 (`OMP_NUM_THREADS=1`).",
          f"- CPU: {manifest['cpu']} ({manifest['cpu_cores']} cores); "
          f"Python {manifest['python']}; numpy {manifest['numpy']}; pandas "
          f"{manifest['pandas']}; scipy {manifest['scipy']}; scikit-learn {manifest['sklearn']}.",
          "- Timing predictions are never used for any metric.",
          "- Aggregate results: `end_to_end_timing.csv`; environment manifest: "
          "`timing_manifest.json`. Raw per-record timing files are intentionally NOT saved "
          "(storage constraint).",
          "",
          "| method | mean ms/record | p50 | p95 | throughput (rec/s) |",
          "|---|---:|---:|---:|---:|"]
    for r in summary_rows:
        md.append(f"| {r['method']} | {r['mean_ms_per_record']:.3f} | "
                  f"{r['median_ms_per_record']:.3f} | {r['p95_ms_per_record']:.3f} | "
                  f"{r['mean_throughput_records_per_second']:.1f} |")
    with open(TIMING_DIR / "timing_protocol.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("stage5 done")


if __name__ == "__main__":
    main()
