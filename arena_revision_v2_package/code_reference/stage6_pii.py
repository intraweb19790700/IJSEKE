"""Stage 6: PII and human-validation status.

- Configured SYNTHETIC PII stress test (kept separate from any independent PII
  challenge set): locked-test records with unchanged text are injected with
  each synthetic PII type (phone / email / patient-ID / SSN, same patterns as
  the workspace redaction rules), then we measure (a) redaction-rule recall,
  (b) the student's pii_risk head recall on the injected view, and (c) the
  pii_risk head none_or_redacted rate on the redacted view. All 5 seeds.
- Independent PII challenge set with human labels: ABSENT -> independent PII
  validation BLOCKED; only an annotation template is created.
- LLM provenance (provider, model/version, dates, settings, prompts, schema,
  retries, adjudication) is extracted only from real files/logs: ABSENT for the
  submitted model. Heuristic/hardcoded labelers are NOT human annotations and
  NOT LLM API logs.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, PII_DIR, LABEL_SPACES, RUN_LABEL,
                    redact_text, write_json, PII_PATTERNS)
from student import TfidfStudent

CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")

PII_TYPES = {
    "phone": ("+1-555-{:04d}", 0),
    "email": ("reviewer{:04d}@example.com", 1),
    "patient_id": ("PID-{:06d}", 2),
    "ssn": ("{:03d}-{:02d}-{:04d}", 3),
}
TEMPLATES = [
    " Contact me at {}.",
    " My phone is {}.",
    " Email: {}.",
    " Patient ID: {}.",
    " SSN {}.",
]


def fmt_pii(kind, i):
    spec, which = PII_TYPES[kind]
    if kind == "phone":
        return spec.format((i * 37) % 10000)
    if kind == "email":
        return spec.format((i * 53) % 10000)
    if kind == "patient_id":
        return spec.format((i * 911) % 1000000)
    return spec.format(100 + (i % 800), 10 + (i % 80), 1000 + (i * 7) % 9000)


def main():
    os.makedirs(PII_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "locked_split.csv"))
    test = df[df["locked_split"] == "test"].reset_index(drop=True)
    unchanged = test[test["original_text"].astype(str) == test["redacted_text"].astype(str)]
    # fixed deterministic subset: up to 100 unchanged records for the stress test
    stress = unchanged.head(100).reset_index(drop=True)
    with open(os.path.join(CKPT_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vecs = pickle.load(f)
    char, word = vecs["char"], vecs["word"]

    def transform(texts):
        texts = pd.Series(texts).astype(str)
        return hstack([char.transform(texts), word.transform(texts)]).astype(np.float32).tocsr()

    with open(os.path.join(MODEL_DIR, "training_logs", "selection_log.json")) as f:
        lam = float(json.load(f)["selected_lambda"])
    seeds = [20260715, 20260716, 20260717, 20260718, 20260719]
    models = []
    for seed in seeds:
        m = TfidfStudent(len(char.vocabulary_) + len(word.vocabulary_), seed)
        m.load(os.path.join(CKPT_DIR, f"student_lambda_{lam}_seed_{seed}.npz"))
        models.append((seed, m))

    base_texts = stress["original_text"].astype(str).tolist()
    rows = []
    for kind in PII_TYPES:
        for i, base in enumerate(base_texts):
            pii = fmt_pii(kind, i)
            injected = base + TEMPLATES[PII_TYPES[kind][1]].format(pii)
            redacted = redact_text(injected)
            rule_removed = pii not in redacted
            Xi = transform([injected])
            Xr = transform([redacted])
            for seed, m in models:
                pi = m.predict_idx(Xi)["pii_risk"][0]
                pr = m.predict_idx(Xr)["pii_risk"][0]
                rows.append({
                    "run_label": RUN_LABEL, "seed": seed, "pii_type": kind,
                    "record_id": stress["record_id"].iloc[i],
                    "injected_pii": pii,
                    "redaction_rule_removed_pii": bool(rule_removed),
                    "pii_head_pred_on_injected": LABEL_SPACES["pii_risk"][int(pi)],
                    "pii_head_pred_on_redacted": LABEL_SPACES["pii_risk"][int(pr)],
                    "pii_head_correct_on_injected": bool(pi == 1),  # contains_pii
                    "pii_head_correct_on_redacted": bool(pr == 0),    # none_or_redacted
                })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PII_DIR, "synthetic_pii_stress_test.csv"), index=False)

    # summary per type and seed
    summ = res.groupby(["pii_type", "seed"]).agg(
        support=("record_id", "count"),
        redaction_rule_recall=("redaction_rule_removed_pii", "mean"),
        pii_head_recall_on_injected=("pii_head_correct_on_injected", "mean"),
        pii_head_none_or_redacted_rate_on_redacted=("pii_head_correct_on_redacted", "mean"),
    ).reset_index()
    summ.to_csv(os.path.join(PII_DIR, "synthetic_pii_stress_test_summary.csv"), index=False)
    overall = res.groupby("seed").agg(
        support=("record_id", "count"),
        redaction_rule_recall=("redaction_rule_removed_pii", "mean"),
        pii_head_recall_on_injected=("pii_head_correct_on_injected", "mean"),
        pii_head_none_or_redacted_rate_on_redacted=("pii_head_correct_on_redacted", "mean"),
    ).reset_index()
    overall.insert(0, "pii_type", "ALL")
    pd.concat([summ, overall]).to_csv(
        os.path.join(PII_DIR, "synthetic_pii_stress_test_summary.csv"), index=False)

    # human annotation template (no real annotations exist)
    template = pd.DataFrame(columns=[
        "record_id", "text_view", "pii_type", "pii_text", "span_start", "span_end",
        "label", "annotator_id", "annotation_date", "adjudicator_id", "notes"])
    example = {c: "" for c in template.columns}
    example.update({"record_id": "EXAMPLE_1", "text_view": "original",
                    "pii_type": "phone", "pii_text": "+1-555-0123",
                    "span_start": "0", "span_end": "0",
                    "label": "contains_pii", "annotator_id": "ANNOTATOR_ID",
                    "annotation_date": "YYYY-MM-DD", "adjudicator_id": "ADJUDICATOR_ID",
                    "notes": "TEMPLATE EXAMPLE - NOT A REAL ANNOTATION"})
    template = pd.concat([template, pd.DataFrame([example])], ignore_index=True)
    template.to_csv(os.path.join(PII_DIR, "human_annotation_template.csv"), index=False)

    # LLM provenance: only from real files/logs
    provenance = {
        "run_label": RUN_LABEL,
        "submitted_model": "RC-EdgeDistill (manuscript IJSEKE-D-26-00185)",
        "llm_provider": "ABSENT - no LLM provider is configured or recorded anywhere in the workspace",
        "llm_model_version": "ABSENT - no teacher/student LLM model version is recorded for the submitted model",
        "llm_dates": "ABSENT - no LLM request dates/logs exist",
        "llm_generation_settings": "ABSENT - no temperature/top_p/max_tokens/seed logs exist",
        "llm_prompts": "ABSENT for the submitted model - the workspace contains Arena task prompts "
                       "(uploads/*.md) and a zero-shot prompt inside src/zero_shot_qwen.py for the "
                       "Qwen prototype, but no prompts or logs for the submitted RC-EdgeDistill model",
        "llm_schema": "ABSENT - no structured-output schema logs for the submitted model",
        "llm_retries": "ABSENT - no retry logs exist",
        "adjudication_logs": "ABSENT for the submitted model - workspace adjudication outputs "
                             "(arena_results/ai_adjudicated_*.csv) are AI-adjudicated weak-supervision "
                             "labels produced by heuristic/hardcoded scripts (src/adjudicate_2000.py, "
                             "src/generate_adjudication.py), not by recorded LLM API calls",
        "human_annotations": "ABSENT - no independent human labels exist; heuristic/hardcoded "
                             "labelers are not human annotations",
        "conclusion": "No LLM provenance or human-validation metadata can be reported for the "
                      "submitted model; nothing was fabricated.",
    }
    write_json(os.path.join(PII_DIR, "llm_provenance_status.json"), provenance)

    md = []
    md.append("# PII and Human-Validation Status")
    md.append("")
    md.append(f"- Run label: {RUN_LABEL}")
    md.append("- **In-distribution pii_risk task metrics** (locked test, redacted view) are reported "
              "separately in `model/locked_test_metrics_by_seed.csv` (pii_risk_macro_f1).")
    md.append("")
    md.append("## Configured synthetic PII stress test (separate from any independent challenge set)")
    md.append("")
    md.append("Method: locked-test records whose original and redacted texts are identical (no PII) "
              "were injected with each synthetic PII type (phone `+1-555-XXXX`, email, `PID-XXXXXX`, "
              "SSN `XXX-XX-XXXX`; 100 records per type; deterministic values). For each injected "
              "record: (a) redaction-rule recall = injected string removed by the local redaction "
              "rules; (b) student pii_risk head recall on the injected view (expected `contains_pii`); "
              "(c) student pii_risk head `none_or_redacted` rate on the redacted view. All 5 seeds.")
    md.append("")
    md.append("Files: `synthetic_pii_stress_test.csv` (per record x seed x type), "
              "`synthetic_pii_stress_test_summary.csv` (per type x seed + overall).")
    md.append("")
    md.append("This stress test uses ONLY configured synthetic PII and the model's own head; it is "
              "kept separate from an independent PII challenge set and is not human validation.")
    md.append("")
    md.append("## Independent PII challenge set / human validation")
    md.append("")
    md.append("- Independent human PII labels: **ABSENT** -> **independent PII validation: BLOCKED**.")
    md.append("- Per-type precision/recall/F1/support against human labels and de-identified false "
              "negatives: **BLOCKED** (no human labels exist; none were fabricated).")
    md.append("- No agreement statistics are reported.")
    md.append("- Only an annotation template was created: `human_annotation_template.csv` "
              "(contains one clearly marked TEMPLATE EXAMPLE row, not a real annotation).")
    md.append("")
    md.append("## LLM provenance (extracted only from real files/logs)")
    md.append("")
    md.append("| Item | Status |")
    md.append("|---|---|")
    md.append("| LLM provider | ABSENT |")
    md.append("| LLM model/version | ABSENT |")
    md.append("| LLM dates | ABSENT |")
    md.append("| LLM generation settings | ABSENT |")
    md.append("| LLM prompts (submitted model) | ABSENT |")
    md.append("| LLM schema | ABSENT |")
    md.append("| LLM retries | ABSENT |")
    md.append("| adjudication logs (submitted model) | ABSENT |")
    md.append("")
    md.append("The workspace's labeling artifacts (teacher labels, AI-adjudicated labels) are "
              "AI-adjudicated weak-supervision labels produced by heuristic/hardcoded scripts; they "
              "are not human annotations and not LLM API logs. See `llm_provenance_status.json`.")
    with open(os.path.join(PII_DIR, "pii_and_human_validation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(summ.to_string(index=False))
    print(overall.to_string(index=False))
    print("stage6 done")


if __name__ == "__main__":
    main()
