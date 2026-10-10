"""Shared constants and utilities for the RC-EdgeDistill major-revision run v2.

Run label: CORRECTED_FAITHFUL_REIMPLEMENTATION.

The exact submitted implementation is NOT available in this workspace, so this
run is a corrected faithful reimplementation of manuscript IJSEKE-D-26-00185
Section 3.3 (see ../implementation_assumptions.md). The reference
``code_reference/student.py`` shipped with the package contains an INCOMPLETE
symmetric-KL backward pass; it is corrected here (see ``student.py`` and
``../gradient_check/``).

Storage-constrained rules honoured by every stage:
- no model checkpoints, no tfidf_vectorizer.pkl, no fitted_models.pkl;
- no per-baseline prediction files, no raw timing files;
- models are fitted and released one at a time, in memory.
"""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (robust to the checkout location)
# ---------------------------------------------------------------------------
SRC_FILE = Path(__file__).resolve()
SRC_DIR = SRC_FILE.parent                       # .../major_revision_run_v2/src
RUN_ROOT = SRC_DIR.parent                       # .../major_revision_run_v2
PKG_ROOT = RUN_ROOT.parent.parent               # .../arena_revision_v2_package
INPUT_DIR = PKG_ROOT / "input"
CODE_REF_DIR = PKG_ROOT / "code_reference"

DATA_DIR = RUN_ROOT / "data"
MODEL_DIR = RUN_ROOT / "model"
BASELINES_DIR = RUN_ROOT / "baselines"
TIMING_DIR = RUN_ROOT / "timing"
PII_DIR = RUN_ROOT / "pii"
REPRO_DIR = RUN_ROOT / "reproducibility"
GRADIENT_CHECK_DIR = RUN_ROOT / "gradient_check"
LOGS_DIR = RUN_ROOT / "logs"

DATA_ORIGINAL = INPUT_DIR / "prepared_paired_dataset_2000.csv"
MANUSCRIPT = INPUT_DIR / "IJSEKE-D-26-00185.md"
DECISION_LETTER = INPUT_DIR / "decision_letter_reviewers.txt"
REVISION_PLAN = INPUT_DIR / "review_revision_plan.md"

REQUIRED_REFERENCE_FILES = [
    "common.py", "metrics_utils.py", "redaction.py", "student.py",
    "stage1_data_audit_split.py", "stage1b_leakage_audit_json.py",
    "stage2_train_student.py", "stage3_test_student.py", "stage4_baselines.py",
    "stage5_timing.py", "stage6_pii.py", "stage7_repro_report.py",
]

EXPECTED_DATA_SHA256 = "bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e"
EXPECTED_ROWS = 2000

RUN_LABEL = "CORRECTED_FAITHFUL_REIMPLEMENTATION"
RUN_DATE = "2026-10-10"

# ---------------------------------------------------------------------------
# Manuscript-specified hyperparameters (Section 3.3 / 4.1)
# ---------------------------------------------------------------------------
CHAR_FEATURES = 6000        # char_wb (2,5) TF-IDF, max 6,000
WORD_FEATURES = 4000        # word (1,2) TF-IDF, max 4,000
SHARED_DIM = 384            # shared projection dimension
DROPOUT = 0.15              # dropout on the shared representation
LR = 0.006                  # Adam learning rate
WEIGHT_DECAY = 1e-4         # Adam weight decay
EPOCHS = 160                # 160 FULL-BATCH epochs (manuscript; NOT 40, NOT mini-batch)
FULL_BATCH = True           # one gradient update per epoch over the whole train split
LAMBDA_GRID = [0.0, 0.03, 0.05, 0.10, 0.25]
SEEDS = [20260715, 20260716, 20260717, 20260718, 20260719]

# Preregistered (recorded, not manuscript-mandated) protocol constants.
SPLIT_SEED = 20260720
NEAR_DUP_THRESHOLD = 0.90
SELECTION_TOLERANCE = 0.005
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260721
BASELINE_SEED = 20260715
LOG_EVERY = 10              # validation logging cadence (epochs)

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
ALL_TASKS = ["service_domain", "sentiment", "priority", "pii_risk"]

# Local deterministic redaction rules (identical to code_reference/redaction.py).
PII_PATTERNS = [
    (re.compile(r"\+1-555-\d{4}"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"PID-\d{6}"), "[PATIENT_ID]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]


def redact_text(text):
    redacted = str(text)
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def contains_pii(text):
    return any(pattern.search(str(text)) for pattern, _ in PII_PATTERNS)


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_to_idx(task, label):
    return LABEL_SPACES[task].index(label)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ensure_dirs():
    for d in [DATA_DIR, MODEL_DIR, BASELINES_DIR, TIMING_DIR, PII_DIR,
              REPRO_DIR, GRADIENT_CHECK_DIR, LOGS_DIR, SRC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
