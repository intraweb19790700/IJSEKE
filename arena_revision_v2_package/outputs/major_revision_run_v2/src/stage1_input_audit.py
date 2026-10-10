"""Stage 1: input verification + leakage audit + group-aware locked split.

PHASE 1 (input check, stops on any failure):
1. prepared_paired_dataset_2000.csv has 2,000 records;
2. its SHA-256 is bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e;
3. IJSEKE-D-26-00185.md, decision_letter_reviewers.txt, review_revision_plan.md readable;
4. all required code_reference Python files are present;
5. data columns and label values are recorded.

Then (before any modeling):
- exact duplicate audit: NFKC + casefold + whitespace collapse on BOTH text
  views; union-find groups;
- near duplicate audit: char_wb (3,5) TF-IDF cosine similarity >= 0.90 on both
  views; union-find groups;
- group-aware locked split (seed 20260720): within each language, groups are
  shuffled, sorted by size descending, and greedily assigned to the split most
  below its target share after assignment (train 70% / validation 15% / test 15%),
  so no duplicate/near-duplicate group crosses splits.

Outputs under ../data/: input_verification.{json,md}, hashes.json,
leakage_audit.{csv,md,json}, locked_split.csv, locked_split_manifest.json.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, DATA_ORIGINAL, MANUSCRIPT, DECISION_LETTER,
                    REVISION_PLAN, CODE_REF_DIR, REQUIRED_REFERENCE_FILES,
                    EXPECTED_DATA_SHA256, EXPECTED_ROWS, LABEL_SPACES,
                    NEAR_DUP_THRESHOLD, SPLIT_SEED, RUN_LABEL, RUN_DATE,
                    PKG_ROOT, RUN_ROOT, normalize_text, sha256_file,
                    write_json, ensure_dirs)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_dup_pairs(texts, threshold):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          dtype=np.float32)
    X = normalize(vec.fit_transform(texts.astype(str)))
    S = (X @ X.T).tocsr()
    S.setdiag(0.0)
    S.eliminate_zeros()
    coo = S.tocoo()
    mask = coo.data >= threshold
    pairs = set()
    for i, j in zip(coo.row[mask].tolist(), coo.col[mask].tolist()):
        if i < j:
            pairs.add((i, j))
    return pairs


def package_tree():
    lines = []
    for p in sorted(PKG_ROOT.rglob("*")):
        if p.is_file():
            lines.append(f"{p.relative_to(PKG_ROOT.parent)}  ({p.stat().st_size} bytes)")
    return lines


def verify_inputs():
    checks = []

    def record(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                       "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    print("=" * 78)
    print("CONNECTED REPOSITORY / PACKAGE LOCATION")
    print("=" * 78)
    print(f"repository root : {PKG_ROOT.parent}")
    print(f"package root    : {PKG_ROOT}")
    print(f"run root        : {RUN_ROOT}")
    print()
    print("=" * 78)
    print("COMPLETE PACKAGE TREE (arena_revision_v2_package/)")
    print("=" * 78)
    for line in package_tree():
        print(line)
    print()

    ok = True
    df = pd.read_csv(DATA_ORIGINAL)
    ok &= record("dataset record count",
                 len(df) == EXPECTED_ROWS,
                 f"{len(df)} records (expected {EXPECTED_ROWS})")
    digest = sha256_file(DATA_ORIGINAL)
    ok &= record("dataset SHA-256",
                 digest == EXPECTED_DATA_SHA256,
                 f"{digest} (expected {EXPECTED_DATA_SHA256})")
    for label, path in [("manuscript IJSEKE-D-26-00185.md", MANUSCRIPT),
                        ("decision_letter_reviewers.txt", DECISION_LETTER),
                        ("review_revision_plan.md", REVISION_PLAN)]:
        readable = path.is_file() and path.stat().st_size > 0
        try:
            with open(path, encoding="utf-8") as f:
                head = f.read(200)
            readable = readable and len(head) > 0
        except OSError:
            readable = False
        ok &= record(f"readable: {label}", readable,
                     f"{path.name}, {path.stat().st_size if path.exists() else 0} bytes")
    missing = [f for f in REQUIRED_REFERENCE_FILES
               if not (CODE_REF_DIR / f).is_file()]
    ok &= record("all required code_reference Python files present",
                 not missing,
                 f"{len(REQUIRED_REFERENCE_FILES) - len(missing)}/{len(REQUIRED_REFERENCE_FILES)}"
                 + (f"; MISSING: {missing}" if missing else ""))
    columns = list(df.columns)
    label_values = {c: sorted(df[c].astype(str).unique().tolist())
                    for c in ["language", "source_family", "split", "service_domain",
                              "sentiment", "priority", "pii_original", "pii_redacted",
                              "confidence"]}
    ok &= record("data columns and label values recorded",
                 True, f"columns={columns}")
    print()
    print("Label value sets:")
    for k, v in label_values.items():
        print(f"  {k}: {v}")
    print()
    return ok, checks, df, columns, label_values, digest


def main():
    ensure_dirs()
    ok, checks, df, columns, label_values, digest = verify_inputs()

    verification = {
        "run_label": RUN_LABEL,
        "date": RUN_DATE,
        "repository_root": str(PKG_ROOT.parent),
        "package_root": str(PKG_ROOT),
        "run_root": str(RUN_ROOT),
        "checks": checks,
        "dataset": {
            "path": str(DATA_ORIGINAL),
            "rows": int(len(df)),
            "sha256": digest,
            "sha256_expected": EXPECTED_DATA_SHA256,
            "sha256_match": digest == EXPECTED_DATA_SHA256,
            "columns": columns,
            "label_values": label_values,
        },
        "overall": "PASS" if ok else "FAIL",
    }
    write_json(DATA_DIR / "input_verification.json", verification)

    md = ["# Stage 1 — Input Verification", "",
          f"- Run label: `{RUN_LABEL}` | Date: {RUN_DATE}",
          f"- Repository root: `{PKG_ROOT.parent}`",
          f"- Package root: `{PKG_ROOT}`",
          f"- Run root: `{RUN_ROOT}`", "",
          "## Checks", ""]
    for c in checks:
        md.append(f"- [{c['status']}] {c['check']}: {c['detail']}")
    md += ["", "## Dataset columns", "", f"`{', '.join(columns)}`", "",
           "## Label values", ""]
    for k, v in label_values.items():
        md.append(f"- `{k}`: {v}")
    md += ["", f"**Overall: {verification['overall']}**",
           "", "If any check fails, the run stops here (no training, no partial outputs)."]
    with open(DATA_DIR / "input_verification.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    if not ok:
        print("INPUT VERIFICATION FAILED - stopping. No training, no partial outputs.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # hashes of all inputs (dataset + manuscript + reference code)
    # ------------------------------------------------------------------
    import hashlib
    ref_hashes = {}
    for p in sorted(CODE_REF_DIR.glob("*.py")):
        ref_hashes[f"code_reference/{p.name}"] = sha256_file(p)
    hashes = {
        "run_label": RUN_LABEL,
        "date": RUN_DATE,
        "input/prepared_paired_dataset_2000.csv": digest,
        "input/IJSEKE-D-26-00185.md": sha256_file(MANUSCRIPT),
        "input/decision_letter_reviewers.txt": sha256_file(DECISION_LETTER),
        "input/review_revision_plan.md": sha256_file(REVISION_PLAN),
        "code_reference_files": ref_hashes,
    }
    write_json(DATA_DIR / "hashes.json", hashes)

    # ------------------------------------------------------------------
    # leakage audit (exact + near duplicates, union-find groups)
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("LEAKAGE AUDIT (exact + near duplicates)")
    print("=" * 78)
    n = len(df)
    uf = UnionFind(n)
    df["norm_original"] = df["original_text"].map(normalize_text)
    df["norm_redacted"] = df["redacted_text"].map(normalize_text)
    exact_pair_count = 0
    for col in ["norm_original", "norm_redacted"]:
        for key, idx in df.groupby(col).groups.items():
            idxs = list(idx)
            exact_pair_count += len(idxs) * (len(idxs) - 1) // 2
            for k in idxs[1:]:
                uf.union(idxs[0], k)
    print(f"exact duplicate pairs (either view): {exact_pair_count}")

    nd_pairs = set()
    for col in ["original_text", "redacted_text"]:
        pairs = near_dup_pairs(df[col], NEAR_DUP_THRESHOLD)
        print(f"near duplicate pairs (cosine >= {NEAR_DUP_THRESHOLD}, view={col}): {len(pairs)}")
        for (i, j) in pairs:
            nd_pairs.add((i, j))
            uf.union(i, j)
    print(f"near duplicate pairs (union of views): {len(nd_pairs)}")

    roots = [uf.find(i) for i in range(n)]
    uniq_roots = sorted(set(roots))
    root_to_gid = {r: k for k, r in enumerate(uniq_roots)}
    df["group_id"] = [root_to_gid[r] for r in roots]
    group_sizes = df.groupby("group_id").size()
    df["group_size"] = df["group_id"].map(group_sizes)

    exact_member = np.zeros(n, dtype=bool)
    for col in ["norm_original", "norm_redacted"]:
        sizes = df.groupby(col)[col].transform("size")
        exact_member |= (sizes > 1).values
    near_member = np.zeros(n, dtype=bool)
    for (i, j) in nd_pairs:
        near_member[i] = True
        near_member[j] = True
    df["exact_dup_member"] = exact_member
    df["near_dup_member"] = near_member
    exact_partners = np.zeros(n, dtype=int)
    for col in ["norm_original", "norm_redacted"]:
        sizes = df.groupby(col)[col].transform("size")
        exact_partners = np.maximum(exact_partners, (sizes - 1).values)
    near_partners = np.zeros(n, dtype=int)
    for (i, j) in nd_pairs:
        near_partners[i] += 1
        near_partners[j] += 1
    df["exact_dup_partners"] = exact_partners
    df["near_dup_partners"] = near_partners

    # ------------------------------------------------------------------
    # group-aware locked split
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"GROUP-AWARE LOCKED SPLIT (seed {SPLIT_SEED})")
    print("=" * 78)
    rng = np.random.default_rng(SPLIT_SEED)
    df["locked_split"] = ""
    split_targets = {"validation": 0.15, "test": 0.15, "train": 0.70}
    for lang, sub in df.groupby("language"):
        gids = sorted(sub["group_id"].unique())
        rng.shuffle(gids)
        gid_size = sub.groupby("group_id").size().to_dict()
        gids.sort(key=lambda g: -gid_size[g])  # largest first (stable after shuffle)
        n_lang = len(sub)
        current = {"train": 0, "validation": 0, "test": 0}
        for g in gids:
            s = gid_size[g]
            best, best_def = None, None
            for sp in ["validation", "test", "train"]:
                deficit = (current[sp] + s) - split_targets[sp] * n_lang
                if best_def is None or deficit < best_def - 1e-9:
                    best, best_def = sp, deficit
            df.loc[df["group_id"] == g, "locked_split"] = best
            current[best] += s

    crossing_groups = int((df.groupby("group_id")["locked_split"].nunique() > 1).sum())

    def separated_pairs(pairs):
        cnt = 0
        for (i, j) in pairs:
            if df.at[i, "locked_split"] != df.at[j, "locked_split"]:
                cnt += 1
        return cnt

    exact_pairs_all = set()
    for col in ["norm_original", "norm_redacted"]:
        for key, idxs in df.groupby(col).groups.items():
            idxs = list(idxs)
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    exact_pairs_all.add((min(idxs[a], idxs[b]), max(idxs[a], idxs[b])))
    exact_sep = separated_pairs(exact_pairs_all)
    near_sep = separated_pairs(nd_pairs)

    split_counts = df["locked_split"].value_counts().to_dict()
    print("locked split counts:", split_counts)
    print(f"groups crossing splits: {crossing_groups} (must be 0)")
    print(f"exact duplicate pairs separated: {exact_sep} (must be 0)")
    print(f"near duplicate pairs separated: {near_sep} (must be 0)")
    if crossing_groups or exact_sep or near_sep:
        print("LEAKAGE AUDIT FAILED - locked split is not group-safe. Stopping.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    audit = df[["record_id", "language", "source_family", "split", "locked_split",
                "group_id", "group_size", "exact_dup_member", "exact_dup_partners",
                "near_dup_member", "near_dup_partners", "pii_original"]].copy()
    audit["changed_text"] = (df["original_text"].astype(str) !=
                             df["redacted_text"].astype(str))
    audit["norm_original_sha256"] = [
        hashlib.sha256(t.encode("utf-8")).hexdigest() for t in df["norm_original"]]
    audit["norm_redacted_sha256"] = [
        hashlib.sha256(t.encode("utf-8")).hexdigest() for t in df["norm_redacted"]]
    audit.to_csv(DATA_DIR / "leakage_audit.csv", index=False)

    locked = df.drop(columns=["norm_original", "norm_redacted"])
    locked.to_csv(DATA_DIR / "locked_split.csv", index=False)

    class_support = {}
    for task, labels in LABEL_SPACES.items():
        col = "pii_original" if task == "pii_risk" else task
        class_support[task] = {
            sp: df[df["locked_split"] == sp][col].value_counts().to_dict()
            for sp in ["train", "validation", "test"]}

    manifest = {
        "run_label": RUN_LABEL,
        "created": RUN_DATE,
        "source_dataset": str(DATA_ORIGINAL),
        "data_sha256": digest,
        "rows": int(n),
        "split_seed": SPLIT_SEED,
        "split_targets": split_targets,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "language_counts_per_split": {
            sp: df[df["locked_split"] == sp]["language"].value_counts().to_dict()
            for sp in ["train", "validation", "test"]},
        "source_family_counts_per_split": {
            sp: df[df["locked_split"] == sp]["source_family"].value_counts().to_dict()
            for sp in ["train", "validation", "test"]},
        "language_source_confounded": bool(
            (pd.crosstab(df["language"], df["source_family"]).gt(0).sum(axis=1) == 1).all()),
        "class_support_per_split": class_support,
        "duplicate_audit": {
            "exact_duplicate_method": "NFKC + casefold + whitespace collapse on both text views; union-find",
            "exact_duplicate_pairs": int(exact_pair_count),
            "exact_duplicate_member_rows": int(exact_member.sum()),
            "near_duplicate_method": f"char_wb (3,5) TF-IDF cosine similarity >= {NEAR_DUP_THRESHOLD} on both views; union-find",
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
            "near_duplicate_pairs": int(len(nd_pairs)),
            "near_duplicate_member_rows": int(near_member.sum()),
            "total_groups": int(df["group_id"].nunique()),
            "largest_group_size": int(group_sizes.max()),
        },
        "crossing_counts": {
            "groups_crossing_splits": crossing_groups,
            "exact_duplicate_pairs_separated": int(exact_sep),
            "near_duplicate_pairs_separated": int(near_sep),
        },
        "metadata_availability": {
            "hospital": "ABSENT - column not present in dataset",
            "website": "ABSENT - column not present in dataset",
            "account": "ABSENT - column not present in dataset",
            "collection_batch": "ABSENT - column not present in dataset",
            "synthetic_parent_id": "ABSENT - no parent-id column; pii_original used as synthetic-injection proxy",
            "available": ["record_id", "language", "source_family", "split (original)",
                          "pii_original", "pii_redacted", "confidence"],
        },
        "text_view_equality": {
            "rows_original_equals_redacted": int((df["original_text"].astype(str) ==
                                                  df["redacted_text"].astype(str)).sum()),
            "rows_changed_by_redaction": int((df["original_text"].astype(str) !=
                                              df["redacted_text"].astype(str)).sum()),
        },
    }
    write_json(DATA_DIR / "locked_split_manifest.json", manifest)

    # machine-readable leakage audit json (derived from saved outputs only)
    group_sizes_audit = audit.groupby("group_id")["group_size"].first()
    audit_doc = {
        "run_label": RUN_LABEL,
        "created": RUN_DATE,
        "data_sha256": digest,
        "rows": int(n),
        "split_seed": SPLIT_SEED,
        "split_targets": split_targets,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "language_counts_per_split": manifest["language_counts_per_split"],
        "source_family_counts_per_split": manifest["source_family_counts_per_split"],
        "language_source_confounded": manifest["language_source_confounded"],
        "duplicate_audit": manifest["duplicate_audit"],
        "group_stats": {
            "total_groups": int(manifest["duplicate_audit"]["total_groups"]),
            "singleton_groups": int((group_sizes_audit == 1).sum()),
            "multi_record_groups": int((group_sizes_audit > 1).sum()),
            "largest_group_size": int(group_sizes_audit.max()),
            "rows_in_multi_record_groups": int(group_sizes_audit[group_sizes_audit > 1].sum()),
        },
        "flag_counts": {
            "exact_duplicate_member_rows": int(audit["exact_dup_member"].sum()),
            "near_duplicate_member_rows": int(audit["near_dup_member"].sum()),
            "rows_with_exact_dup_partners": int((audit["exact_dup_partners"] > 0).sum()),
            "rows_with_near_dup_partners": int((audit["near_dup_partners"] > 0).sum()),
            "changed_text_rows": int(audit["changed_text"].sum()),
            "unchanged_text_rows": int((~audit["changed_text"]).sum()),
        },
        "crossing_counts": manifest["crossing_counts"],
        "metadata_availability": manifest["metadata_availability"],
        "text_view_equality": manifest["text_view_equality"],
        "class_support_per_split": manifest["class_support_per_split"],
        "methods": {
            "exact_duplicates": manifest["duplicate_audit"]["exact_duplicate_method"],
            "near_duplicates": manifest["duplicate_audit"]["near_duplicate_method"],
            "near_duplicate_threshold": manifest["duplicate_audit"]["near_duplicate_threshold"],
            "split_algorithm": f"within each language, groups shuffled with numpy.default_rng({SPLIT_SEED}), "
                               "sorted by size descending, greedily assigned to the split most below "
                               "its target share after assignment (train 70% / validation 15% / test 15%)",
        },
        "limitations": [
            "hospital/website/account/collection-batch/synthetic-parent metadata are ABSENT; "
            "no source-level grouping beyond the duplicate/near-duplicate lineage groups is possible",
            "language and source_family are perfectly confounded (en <-> "
            "english_public_hospital_reviews, zh <-> chinese_public_web_deidentified); "
            "subgroup results are a joint language/source diagnostic only",
        ],
    }
    write_json(DATA_DIR / "leakage_audit.json", audit_doc)

    # human-readable audit
    md = []
    md.append("# Leakage Audit and Group-Aware Locked Split")
    md.append("")
    md.append(f"- Run label: `{RUN_LABEL}` (corrected faithful reimplementation; the exact "
              "submitted implementation is unavailable)")
    md.append(f"- Source dataset (preserved, unmodified): `{DATA_ORIGINAL}`")
    md.append(f"- Data SHA-256: `{digest}` (verified in stage 1)")
    md.append(f"- Rows: {n}; fixed split seed: {SPLIT_SEED}")
    md.append("")
    md.append("## Exact duplicate audit")
    md.append("")
    md.append("Method: NFKC normalization + case folding + whitespace collapse, applied to "
              "BOTH `original_text` and `redacted_text`; records sharing a normalized text in "
              "either view are unioned into one group.")
    md.append("")
    md.append(f"- Exact-duplicate pairs: **{exact_pair_count}**")
    md.append(f"- Rows that are members of an exact-duplicate group: **{int(exact_member.sum())}**")
    md.append("")
    md.append("## Near-duplicate audit")
    md.append("")
    md.append(f"Method: char_wb (3,5) TF-IDF vectors (min_df=2), cosine similarity, fixed "
              f"threshold **{NEAR_DUP_THRESHOLD}**; pairs at or above the threshold are unioned "
              "(computed on both views).")
    md.append("")
    md.append(f"- Near-duplicate pairs (either view): **{len(nd_pairs)}**")
    md.append(f"- Rows that are members of a near-duplicate pair: **{int(near_member.sum())}**")
    md.append(f"- Total lineage groups (exact + near, union-find): "
              f"**{int(df['group_id'].nunique())}** (largest group size {int(group_sizes.max())})")
    md.append("")
    md.append("## Metadata audit and limitations")
    md.append("")
    md.append("| Metadata | Availability |")
    md.append("|---|---|")
    md.append("| hospital | ABSENT (column not present) |")
    md.append("| website | ABSENT (column not present) |")
    md.append("| account | ABSENT (column not present) |")
    md.append("| collection batch | ABSENT (column not present) |")
    md.append("| synthetic-parent id | ABSENT (no parent-id column; `pii_original` used as "
              "synthetic-injection proxy) |")
    md.append("| language / source_family | present; **perfectly confounded** "
              "(en <-> english_public_hospital_reviews, zh <-> chinese_public_web_deidentified) |")
    md.append("")
    md.append("Because language and source_family are perfectly confounded, they are used for "
              "**stratification** only, not as grouping constraints (a source-level grouping "
              "would be degenerate: one group per language). All subgroup results are reported "
              "as a **joint language/source diagnostic**; no separate causal language or source "
              "effects are claimed.")
    md.append("")
    md.append("## Group-aware locked split")
    md.append("")
    md.append(f"Algorithm: within each language, groups shuffled with `numpy.default_rng({SPLIT_SEED})`, "
              "sorted by size descending, greedily assigned to the split most below its target "
              "share after assignment (train 70% / validation 15% / test 15%).")
    md.append("")
    md.append("| Split | Rows | en | zh |")
    md.append("|---|---:|---:|---:|")
    for sp in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == sp]
        md.append(f"| {sp} | {len(sub)} | {int((sub['language'] == 'en').sum())} | "
                  f"{int((sub['language'] == 'zh').sum())} |")
    md.append("")
    md.append("## Leakage crossing checks")
    md.append("")
    md.append(f"- Groups crossing splits: **{crossing_groups}** (must be 0)")
    md.append(f"- Exact-duplicate pairs separated across splits: **{exact_sep}** (must be 0)")
    md.append(f"- Near-duplicate pairs separated across splits: **{near_sep}** (must be 0)")
    md.append("")
    md.append("The original `split` column is preserved in `locked_split.csv` for reference; all "
              "downstream stages use `locked_split`. Locked-test labels were not used for any "
              "model selection.")
    with open(DATA_DIR / "leakage_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print()
    print("stage1 done: input verification PASS, leakage audit PASS, locked split written")


if __name__ == "__main__":
    main()
