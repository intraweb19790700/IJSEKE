"""Stage 6: synthetic PII stress test + human-validation / LLM-provenance status.

- Configured SYNTHETIC PII stress test (kept separate from human validation):
  locked-test records whose text is unchanged by redaction are injected with
  each configured synthetic PII type (phone / email / patient-ID / SSN — the
  same pattern families as the local redaction rules), then we measure
  (a) redaction-rule recall, (b) the student's pii_risk head recall on the
  injected view, and (c) the pii_risk head none_or_redacted rate on the
  redacted view. All 5 seeds (each model refit in memory, then released).
- Storage rule: only the SUMMARY CSV is saved (no per-record stress-test file).
- Independent PII challenge set with human labels: ABSENT -> independent PII
  validation BLOCKED; only an annotation template is created.
- LLM provenance (provider, model/version, dates, settings, prompts, schema,
  retries, adjudication logs) is extracted only from real files/logs: ABSENT
  for the submitted model. Nothing is fabricated.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, MODEL_DIR, PII_DIR, LOGS_DIR, LABEL_SPACES, SEEDS,
                    RUN_LABEL, RUN_DATE, redact_text, write_json, read_json,
                    ensure_dirs)
from stage2_train_student import build_features, build_targets, train_one

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
STRESS_RECORDS = 100  # unchanged locked-test records per PII type


def fmt_pii(kind, i):
    spec, _ = PII_TYPES[kind]
    if kind == "phone":
        return spec.format((i * 37) % 10000)
    if kind == "email":
        return spec.format((i * 53) % 10000)
    if kind == "patient_id":
        return spec.format((i * 911) % 1000000)
    return spec.format(100 + (i % 800), 10 + (i % 80), 1000 + (i * 7) % 9000)


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_DIR / "locked_split.csv")
    test = df[df["locked_split"] == "test"].reset_index(drop=True)
    unchanged = test[test["original_text"].astype(str) ==
                     test["redacted_text"].astype(str)]
    stress = unchanged.head(STRESS_RECORDS).reset_index(drop=True)
    print(f"stress-test base records (unchanged by redaction): {len(stress)}")

    feats, feature_dim, char_n, word_n = build_features(df)
    feats["feature_dim"] = feature_dim
    targets = build_targets(df)

    selection = read_json(MODEL_DIR / "selection_log.json")
    lam = float(selection["selected_lambda"])

    from sklearn.feature_extraction.text import TfidfVectorizer
    train = df[df["locked_split"] == "train"]
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           max_features=6000, min_df=2, dtype=np.float32)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=4000, min_df=2, lowercase=True,
                           dtype=np.float32)
    fit_text = pd.concat([train["original_text"], train["redacted_text"]]).astype(str)
    char.fit(fit_text)
    word.fit(fit_text)

    def transform(texts):
        texts = pd.Series(texts).astype(str)
        return hstack([char.transform(texts),
                       word.transform(texts)]).astype(np.float32).tocsr()

    base_texts = stress["original_text"].astype(str).tolist()
    rows = []
    for seed in SEEDS:
        # refit the selected model in memory (no checkpoint saved/loaded)
        log_path = LOGS_DIR / "stage2_training" / f"pii_retrain_lambda_{lam}_seed_{seed}.log"
        model, _, _, _ = train_one(lam, seed, feats, targets, 160, log_path)
        for kind in PII_TYPES:
            for i, base in enumerate(base_texts):
                pii = fmt_pii(kind, i)
                injected = base + TEMPLATES[PII_TYPES[kind][1]].format(pii)
                redacted = redact_text(injected)
                rule_removed = pii not in redacted
                Xi = transform([injected])
                Xr = transform([redacted])
                pi = int(model.predict_idx(Xi)["pii_risk"][0])
                pr = int(model.predict_idx(Xr)["pii_risk"][0])
                rows.append({
                    "run_label": RUN_LABEL, "seed": seed, "pii_type": kind,
                    "record_id": stress["record_id"].iloc[i],
                    "redaction_rule_removed_pii": bool(rule_removed),
                    "pii_head_pred_on_injected": LABEL_SPACES["pii_risk"][pi],
                    "pii_head_pred_on_redacted": LABEL_SPACES["pii_risk"][pr],
                    "pii_head_correct_on_injected": bool(pi == 1),   # contains_pii
                    "pii_head_correct_on_redacted": bool(pr == 0),   # none_or_redacted
                })
        del model  # release immediately
        print(f"seed {seed} stress test done ({len(rows)} rows so far)")

    res = pd.DataFrame(rows)

    # summary per type and seed
    summ = res.groupby(["pii_type", "seed"]).agg(
        support=("record_id", "count"),
        redaction_rule_recall=("redaction_rule_removed_pii", "mean"),
        pii_head_recall_on_injected=("pii_head_correct_on_injected", "mean"),
        pii_head_none_or_redacted_rate_on_redacted=("pii_head_correct_on_redacted", "mean"),
    ).reset_index()
    overall = res.groupby("seed").agg(
        support=("record_id", "count"),
        redaction_rule_recall=("redaction_rule_removed_pii", "mean"),
        pii_head_recall_on_injected=("pii_head_correct_on_injected", "mean"),
        pii_head_none_or_redacted_rate_on_redacted=("pii_head_correct_on_redacted", "mean"),
    ).reset_index()
    overall.insert(0, "pii_type", "ALL")
    pd.concat([summ, overall]).to_csv(
        PII_DIR / "synthetic_pii_stress_test_summary.csv", index=False)

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
    template.to_csv(PII_DIR / "human_annotation_template.csv", index=False)

    # LLM provenance: only from real files/logs (nothing fabricated)
    provenance = {
        "run_label": RUN_LABEL,
        "date": RUN_DATE,
        "submitted_model": "RC-EdgeDistill (manuscript IJSEKE-D-26-00185)",
        "llm_provider": "ABSENT - no LLM provider is configured or recorded anywhere in the package",
        "llm_model_version": "ABSENT - no LLM model name/version is recorded for label construction",
        "llm_dates": "ABSENT - no LLM request dates/logs exist",
        "llm_generation_settings": "ABSENT - no temperature/top_p/max_tokens/seed logs exist",
        "llm_prompts": "ABSENT - the two labeling instructions gA/gB are described in the "
                       "manuscript but the exact prompts are not present in the package",
        "llm_schema": "ABSENT - no structured-output schema logs exist",
        "llm_retries": "ABSENT - no retry logs exist",
        "adjudication_logs": "ABSENT - the adjudication step a(.) is described procedurally "
                             "in the manuscript; no adjudication records/logs are present",
        "human_annotations": "ABSENT - no independent human labels exist anywhere in the package",
        "human_agreement_statistics": "BLOCKED - cannot be computed without human labels",
        "ethics_records": "ABSENT - no ethics approval/exemption records are present in the package",
        "conclusion": "No LLM provenance, human-validation, or ethics metadata can be reported "
                      "for the submitted model; nothing was fabricated.",
    }
    write_json(PII_DIR / "llm_provenance_status.json", provenance)

    md = []
    md.append("# PII and Human-Validation Status")
    md.append("")
    md.append(f"- Run label: `{RUN_LABEL}` | Date: {RUN_DATE}")
    md.append("- **In-distribution pii_risk task metrics** (locked test, redacted view) are "
              "reported separately in `model/locked_test_metrics_by_seed.csv` "
              "(`pii_risk_macro_f1`).")
    md.append("")
    md.append("## Configured synthetic PII stress test (separate from human validation)")
    md.append("")
    md.append(f"Method: locked-test records whose original and redacted texts are identical "
              f"(no configured PII) — the first {STRESS_RECORDS} such records in file order — "
              "were injected with each configured synthetic PII type (phone `+1-555-XXXX`, "
              "email, `PID-XXXXXX`, SSN `XXX-XX-XXXX`; deterministic values). For each "
              "injected record: (a) redaction-rule recall = injected string removed by the "
              "local redaction rules; (b) student pii_risk head recall on the injected view "
              "(expected `contains_pii`); (c) student pii_risk head `none_or_redacted` rate "
              f"on the redacted view. All {len(SEEDS)} seeds (each model refit in memory).")
    md.append("")
    md.append("This stress test uses ONLY configured synthetic PII and the model's own head; "
              "it is kept separate from an independent PII challenge set and is NOT human "
              "validation. Only the summary is saved (`synthetic_pii_stress_test_summary.csv`); "
              "no per-record stress-test file is persisted (storage constraint).")
    md.append("")
    md.append("| pii_type | seed | support | rule recall | pii-head recall (injected) | "
              "pii-head none_or_redacted (redacted) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in pd.concat([summ, overall]).iterrows():
        md.append(f"| {r['pii_type']} | {int(r['seed'])} | {int(r['support'])} | "
                  f"{r['redaction_rule_recall']:.4f} | "
                  f"{r['pii_head_recall_on_injected']:.4f} | "
                  f"{r['pii_head_none_or_redacted_rate_on_redacted']:.4f} |")
    md.append("")
    md.append("## Independent PII challenge set / human validation")
    md.append("")
    md.append("- Independent human PII labels: **ABSENT** -> **independent PII validation: BLOCKED**.")
    md.append("- Per-type precision/recall/F1/support against human labels and de-identified "
              "false negatives: **BLOCKED** (no human labels exist; none were fabricated).")
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
    md.append("| LLM prompts | ABSENT |")
    md.append("| LLM schema | ABSENT |")
    md.append("| LLM retries | ABSENT |")
    md.append("| adjudication logs | ABSENT |")
    md.append("| human annotations | ABSENT |")
    md.append("| human agreement statistics | BLOCKED |")
    md.append("| ethics records | ABSENT |")
    md.append("")
    md.append("The dataset labels are LLM-assisted weak supervision as described in the "
              "manuscript, but no LLM provider/version/settings/prompts/logs, no independent "
              "human labels, and no ethics records are present in the package. See "
              "`llm_provenance_status.json`. Nothing was fabricated.")
    with open(PII_DIR / "pii_and_human_validation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(summ.to_string(index=False))
    print(overall.to_string(index=False))
    print("stage6 done")


if __name__ == "__main__":
    main()
