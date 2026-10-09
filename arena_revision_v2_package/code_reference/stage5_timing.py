"""Stage 5: common end-to-end CPU timing for RC-EdgeDistill and all runnable
baselines.

Identical for every method: CPU, thread count (OMP_NUM_THREADS=1), software
versions, record order (locked-test file order), batch policy (per-record,
batch=1), warm-up (1 full pass), repetitions (5), timer (time.perf_counter).
Timed pipeline stages per record: input normalization (NFKC + whitespace
collapse), local redaction (regex rules), TF-IDF construction (fitted
vectorizer transform), inference, decoding (argmax -> label), output
serialization (JSON). Model-only latency is NOT mixed in: every timed record
runs the full path. Timing predictions are not used for any metric.
"""
import json
import os
import pickle
import platform
import sys
import time
import unicodedata
import re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, BASELINES_DIR, TIMING_DIR,
                    LABEL_SPACES, CONSISTENCY_TASKS, RUN_LABEL,
                    redact_text, sha256_file, code_hash)
from student import TfidfStudent

CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
WARMUP = 1
REPS = 5
TIMING_SEED_MODEL = 20260715  # student checkpoint and baseline fits for timing


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
    os.makedirs(TIMING_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "locked_split.csv"))
    test = df[df["locked_split"] == "test"].reset_index(drop=True)
    raw_texts = test["original_text"].astype(str).tolist()

    with open(os.path.join(CKPT_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vecs = pickle.load(f)
    char, word = vecs["char"], vecs["word"]

    def tfidf(text):
        from scipy.sparse import csr_matrix, hstack
        return hstack([char.transform([text]), word.transform([text])]).astype(np.float32).tocsr()

    with open(os.path.join(MODEL_DIR, "training_logs", "selection_log.json")) as f:
        lam = float(json.load(f)["selected_lambda"])
    student = TfidfStudent(len(char.vocabulary_) + len(word.vocabulary_), TIMING_SEED_MODEL)
    student.load(os.path.join(CKPT_DIR, f"student_lambda_{lam}_seed_{TIMING_SEED_MODEL}.npz"))

    with open(os.path.join(BASELINES_DIR, "fitted_models.pkl"), "rb") as f:
        fitted = pickle.load(f)  # name -> list per seed of {task: clf}
    baseline_models = {name: models[0] for name, models in fitted.items()}  # seed 20260715

    def norm(text):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

    def student_pipeline(text):
        t = norm(text)
        t = redact_text(t)
        X = tfidf(t)
        logits, _, _, _ = student.forward(X, train=False)
        out = {}
        for task in LABEL_SPACES:
            out[task] = LABEL_SPACES[task][int(logits[task].argmax(axis=1)[0])]
        return json.dumps(out, ensure_ascii=False)

    def make_baseline_pipeline(models):
        def pipeline(text):
            t = norm(text)
            t = redact_text(t)
            X = tfidf(t)
            out = {}
            for task in CONSISTENCY_TASKS:
                out[task] = str(models[task].predict(X)[0])
            return json.dumps(out, ensure_ascii=False)
        return pipeline

    pipelines = {f"rc_edgedistill_lambda_{lam}": student_pipeline}
    for name, models in baseline_models.items():
        pipelines[name] = make_baseline_pipeline(models)

    raw_rows = []
    per_record_rows = []
    summary_rows = []
    for method, fn in pipelines.items():
        # warm-up
        for text in raw_texts:
            fn(text)
        rep_records = []
        for rep in range(REPS):
            lat = []
            t0 = time.perf_counter()
            for i, text in enumerate(raw_texts):
                s = time.perf_counter()
                fn(text)
                lat.append((time.perf_counter() - s) * 1000.0)
                per_record_rows.append({"method": method, "repetition": rep,
                                        "record_index": i, "latency_ms": lat[-1]})
            total = time.perf_counter() - t0
            lat = np.array(lat)
            raw_rows.append({"method": method, "repetition": rep, "warmup": False,
                             "records": len(raw_texts), "total_seconds": total,
                             "mean_ms_per_record": float(lat.mean()),
                             "std_ms_per_record": float(lat.std()),
                             "min_ms_per_record": float(lat.min()),
                             "max_ms_per_record": float(lat.max())})
            rep_records.append(lat.mean())
            print(f"{method} rep={rep} mean={lat.mean():.3f} ms/record total={total:.2f}s")
        rep_records = np.array(rep_records)
        all_lat = np.array([r["latency_ms"] for r in per_record_rows if r["method"] == method])
        summary_rows.append({
            "method": method, "records": len(raw_texts),
            "mean_ms_per_record": float(all_lat.mean()),
            "std_ms_per_record": float(all_lat.std()),
            "median_ms_per_record": float(np.median(all_lat)),
            "min_ms_per_record": float(all_lat.min()),
            "max_ms_per_record": float(all_lat.max()),
            "mean_of_repetition_means_ms": float(rep_records.mean()),
            "std_of_repetition_means_ms": float(rep_records.std()),
            "repetitions": REPS, "warmup_passes": WARMUP,
            "batch_policy": "per-record (batch=1)",
            "threads": 1, "timer": "time.perf_counter",
        })

    pd.DataFrame(raw_rows).to_csv(os.path.join(TIMING_DIR, "raw_timings.csv"), index=False)
    pd.DataFrame(per_record_rows).to_csv(os.path.join(TIMING_DIR, "per_record_latencies.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(TIMING_DIR, "end_to_end_timing.csv"), index=False)

    import sklearn, scipy
    manifest = {
        "run_label": RUN_LABEL,
        "date": "2026-10-09",
        "scope": "end-to-end per-record serving latency (input normalization -> local redaction -> TF-IDF construction -> inference -> decoding -> JSON serialization)",
        "not_included": "model fitting/training time; vectorizer fitting; accuracy metrics (timing predictions are not used for metrics)",
        "record_order": "locked_split.csv locked-test rows in file order",
        "records": len(raw_texts),
        "batch_policy": "per-record (batch=1)",
        "warmup_passes": WARMUP,
        "repetitions": REPS,
        "timer": "time.perf_counter",
        "threads": "OMP_NUM_THREADS=1 (also pinned during training)",
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "input_normalization": "NFKC + whitespace collapse (identical for all methods)",
        "redaction": "regex rules from common.py (identical to workspace src/redaction.py synthetic patterns)",
        "student_checkpoint": f"student_lambda_{lam}_seed_{TIMING_SEED_MODEL}.npz (selected lambda {lam})",
        "baseline_fits": "selected hyperparameters, seed 20260715",
        "data_sha256": sha256_file(os.path.join(DATA_DIR, "locked_split.csv")),
        "code_hash": code_hash([os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
                                 for f in ["common.py", "metrics_utils.py", "student.py",
                                           "stage1_data_audit_split.py", "stage2_train_student.py",
                                           "stage3_test_student.py", "stage4_baselines.py",
                                           "stage5_timing.py"]]),
    }
    with open(os.path.join(TIMING_DIR, "timing_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("stage5 done")


if __name__ == "__main__":
    main()
