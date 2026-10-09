"""Shared constants and utilities for the RC-EdgeDistill major revision run.

Run label: FAITHFUL_REIMPLEMENTATION (the exact submitted implementation is
unavailable in the workspace; see revision_runs/02_code_export/code_mapping.md).
"""
import hashlib
import json
import os
import re
import unicodedata

import numpy as np

RUN_ROOT = "/home/user/revision_runs/major_revision_run"
DATA_DIR = os.path.join(RUN_ROOT, "data")
MODEL_DIR = os.path.join(RUN_ROOT, "model")
BASELINES_DIR = os.path.join(RUN_ROOT, "baselines")
TIMING_DIR = os.path.join(RUN_ROOT, "timing")
PII_DIR = os.path.join(RUN_ROOT, "pii")
REPRO_DIR = os.path.join(RUN_ROOT, "reproducibility")
SRC_DIR = os.path.join(RUN_ROOT, "src")

DATA_WORKING_COPY = os.path.join(DATA_DIR, "prepared_paired_dataset_2000.csv")
DATA_ORIGINAL = "/home/user/uploads/prepared_paired_dataset_2000.csv"

RUN_LABEL = "FAITHFUL_REIMPLEMENTATION"
RUN_DATE = "2026-10-09"

# Manuscript-specified hyperparameters (Section 3.3 TF-IDF student).
CHAR_FEATURES = 6000
WORD_FEATURES = 4000
SHARED_DIM = 384
DROPOUT = 0.15
LR = 0.006
WEIGHT_DECAY = 1e-4
LAMBDA_GRID = [0.0, 0.03, 0.05, 0.10, 0.25]
SEEDS = [20260715, 20260716, 20260717, 20260718, 20260719]

# Assumed (unspecified) details - recorded in model/implementation_assumptions.md.
EPOCHS = 40
BATCH_SIZE = 256
SPLIT_SEED = 20260720
NEAR_DUP_THRESHOLD = 0.90
SELECTION_TOLERANCE = 0.005
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260721
BASELINE_SEED = 20260715

LABEL_SPACES = {
    "service_domain": ["wait_time", "staff_communication", "clinical_process",
                       "billing_insurance", "medication", "environment_facility",
                       "administration", "privacy_security", "praise_general",
                       "other_uncertain"],
    "sentiment": ["positive", "neutral", "negative"],
    "priority": ["low", "medium", "high"],
    "pii_risk": ["none_or_redacted", "contains_pii"],
}
CONSISTENCY_TASKS = ["service_domain", "sentiment", "priority"]
SEMANTIC_TASKS = ["service_domain", "sentiment", "priority"]

# Local redaction rules (identical to workspace src/redaction.py synthetic patterns).
PII_PATTERNS = [
    (re.compile(r"\+1-555-\d{4}"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"PID-\d{6}"), "[PATIENT_ID]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]


def redact_text(text):
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash(paths):
    """SHA-256 over 'relpath:sha256' lines for the given files (sorted)."""
    lines = []
    for p in sorted(paths):
        lines.append(f"{p}:{sha256_file(p)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def normalize_text(text):
    """NFKC + case folding + whitespace collapse (leakage-audit normalization)."""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.casefold()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def label_to_idx(task, label):
    return LABEL_SPACES[task].index(label)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
