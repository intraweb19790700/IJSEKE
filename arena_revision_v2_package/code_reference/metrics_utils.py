"""Metric helpers: macro-F1, per-class metrics, consistency, subgroups, bootstrap."""
import numpy as np

from common import LABEL_SPACES, CONSISTENCY_TASKS


def idx_to_labels(task, idx):
    """Map integer class indices to label strings for task."""
    arr = np.array(LABEL_SPACES[task])
    return arr[np.asarray(idx)]


def macro_f1(y_true, y_pred, labels):
    """Macro-F1 over the given label list (zero_division=0)."""
    f1s = []
    for lab in labels:
        tp = int(np.sum((y_pred == lab) & (y_true == lab)))
        fp = int(np.sum((y_pred == lab) & (y_true != lab)))
        fn = int(np.sum((y_pred != lab) & (y_true == lab)))
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def per_class_metrics(y_true, y_pred, labels):
    rows = []
    for lab in labels:
        tp = int(np.sum((y_pred == lab) & (y_true == lab)))
        fp = int(np.sum((y_pred == lab) & (y_true != lab)))
        fn = int(np.sum((y_pred != lab) & (y_true == lab)))
        support = int(np.sum(y_true == lab))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"class": lab, "precision": prec, "recall": rec,
                     "f1": f1, "support": support})
    return rows


def task_macro_f1s(y_true_by_task, y_pred_by_task):
    out = {}
    for task in LABEL_SPACES:
        out[task] = macro_f1(y_true_by_task[task], y_pred_by_task[task],
                             LABEL_SPACES[task])
    return out


def composite_macro_f1(task_f1s):
    """Composite = mean of the three semantic-task macro-F1 values."""
    return float(np.mean([task_f1s[t] for t in CONSISTENCY_TASKS]))


def consistency_rates(df, pred_col_orig, pred_col_redact, tasks=CONSISTENCY_TASKS):
    """Agreement between original-view and redacted-view predictions."""
    changed = df["original_text"].astype(str) != df["redacted_text"].astype(str)
    out = {}
    for task in tasks:
        agree = (df[f"{pred_col_orig}_{task}"] == df[f"{pred_col_redact}_{task}"])
        out[f"{task}_consistency_all"] = float(agree.mean())
        out[f"{task}_consistency_changed"] = float(agree[changed].mean()) if changed.any() else float("nan")
        out[f"{task}_consistency_unchanged"] = float(agree[~changed].mean()) if (~changed).any() else float("nan")
    out["mean_consistency_all"] = float(np.mean([out[f"{t}_consistency_all"] for t in tasks]))
    out["mean_consistency_changed"] = float(np.mean([out[f"{t}_consistency_changed"] for t in tasks]))
    out["mean_consistency_unchanged"] = float(np.mean([out[f"{t}_consistency_unchanged"] for t in tasks]))
    return out


def bootstrap_ci(y_true, y_pred, labels, metric_fn, n_replicates, seed):
    """Percentile bootstrap CI of a metric over test records."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    for _ in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx], labels))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals
