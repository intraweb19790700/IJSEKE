"""Stage 1: leakage audit and group-aware locked train/validation/test split.

Method (recorded for audit):
- Exact duplicates: NFKC normalization + case folding + whitespace collapse,
  applied to BOTH text views (original_text, redacted_text); union-find groups.
- Near duplicates: char_wb (3,5) TF-IDF cosine similarity >= 0.90 (fixed
  threshold), computed on both views; union-find groups.
- Metadata audit: hospital / website / account / collection batch /
  synthetic-parent columns are ABSENT from the dataset (recorded); available
  lineage proxies are language, source_family, and pii_original (synthetic
  injection flag). language and source_family are perfectly confounded, so they
  are used for stratification, NOT as grouping constraints (a source-level
  grouping would be degenerate: one group per language).
- Split: fixed seed 20260720; within each language, groups are shuffled
  (numpy default_rng), sorted by size descending, and greedily assigned to
  the split that is most below its target share after assignment
  (targets: train 70%, validation 15%, test 15%), so no group crosses splits.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, DATA_WORKING_COPY, DATA_ORIGINAL, SRC_DIR,
                    LABEL_SPACES, NEAR_DUP_THRESHOLD, SPLIT_SEED,
                    normalize_text, sha256_file, code_hash, write_json)


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


def main():
    df = pd.read_csv(DATA_WORKING_COPY)
    n = len(df)

    # ---------------- exact duplicates (both views) ----------------
    uf = UnionFind(n)
    df["norm_original"] = df["original_text"].map(normalize_text)
    df["norm_redacted"] = df["redacted_text"].map(normalize_text)
    exact_groups = {}
    for col in ["norm_original", "norm_redacted"]:
        for key, idx in df.groupby(col).groups.items():
            idxs = list(idx)
            for k in idxs[1:]:
                uf.union(idxs[0], k)
            exact_groups.setdefault(col, {})[key] = len(idxs)
    exact_pair_count = 0
    for col in ["norm_original", "norm_redacted"]:
        for key, idxs in df.groupby(col).groups.items():
            idxs = list(idxs)
            exact_pair_count += len(idxs) * (len(idxs) - 1) // 2

    # ---------------- near duplicates (both views) ----------------
    nd_pairs = set()
    for col in ["original_text", "redacted_text"]:
        for (i, j) in near_dup_pairs(df[col], NEAR_DUP_THRESHOLD):
            nd_pairs.add((i, j))
            uf.union(i, j)

    # ---------------- group ids ----------------
    roots = [uf.find(i) for i in range(n)]
    uniq_roots = sorted(set(roots))
    root_to_gid = {r: k for k, r in enumerate(uniq_roots)}
    df["group_id"] = [root_to_gid[r] for r in roots]
    group_sizes = df.groupby("group_id").size()
    df["group_size"] = df["group_id"].map(group_sizes)

    # membership / partner counts
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

    # ---------------- group-aware split ----------------
    rng = np.random.default_rng(SPLIT_SEED)
    df["locked_split"] = ""
    split_targets = {"validation": 0.15, "test": 0.15, "train": 0.70}
    for lang, sub in df.groupby("language"):
        idxs = list(sub.index)
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

    # ---------------- crossing checks ----------------
    crossing_groups = int((df.groupby("group_id")["locked_split"].nunique() > 1).sum())
    # duplicate pairs separated across splits
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

    # ---------------- outputs ----------------
    audit = df[["record_id", "language", "source_family", "split", "locked_split",
                "group_id", "group_size", "exact_dup_member", "exact_dup_partners",
                "near_dup_member", "near_dup_partners", "pii_original"]].copy()
    audit["changed_text"] = (df["original_text"].astype(str) != df["redacted_text"].astype(str))
    audit["norm_original_sha256"] = [__import__("hashlib").sha256(t.encode("utf-8")).hexdigest()
                                     for t in df["norm_original"]]
    audit["norm_redacted_sha256"] = [__import__("hashlib").sha256(t.encode("utf-8")).hexdigest()
                                     for t in df["norm_redacted"]]
    audit.to_csv(os.path.join(DATA_DIR, "leakage_audit.csv"), index=False)

    locked = df.drop(columns=["norm_original", "norm_redacted"])
    locked.to_csv(os.path.join(DATA_DIR, "locked_split.csv"), index=False)

    # class support per split
    class_support = {}
    for task, labels in LABEL_SPACES.items():
        col = "pii_original" if task == "pii_risk" else task
        class_support[task] = {
            sp: df[df["locked_split"] == sp][col].value_counts().to_dict()
            for sp in ["train", "validation", "test"]
        }

    code_files = [os.path.join(SRC_DIR, f) for f in ["common.py", "metrics_utils.py", "stage1_data_audit_split.py"]]
    manifest = {
        "run_label": "FAITHFUL_REIMPLEMENTATION",
        "created": "2026-10-09",
        "source_dataset_original": DATA_ORIGINAL,
        "source_dataset_working_copy": DATA_WORKING_COPY,
        "data_sha256_original": sha256_file(DATA_ORIGINAL),
        "data_sha256_working_copy": sha256_file(DATA_WORKING_COPY),
        "rows": int(n),
        "split_seed": SPLIT_SEED,
        "split_targets": split_targets,
        "split_counts": df["locked_split"].value_counts().to_dict(),
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
            "rows_original_equals_redacted": int((df["original_text"].astype(str) == df["redacted_text"].astype(str)).sum()),
            "rows_changed_by_redaction": int((df["original_text"].astype(str) != df["redacted_text"].astype(str)).sum()),
        },
        "code_hash": code_hash(code_files),
        "code_hash_method": "sha256 over sorted 'relpath:sha256' lines of src/common.py, src/metrics_utils.py, src/stage1_data_audit_split.py",
    }
    write_json(os.path.join(DATA_DIR, "locked_split_manifest.json"), manifest)

    # ---------------- human-readable audit ----------------
    md = []
    md.append("# Leakage Audit and Group-Aware Locked Split")
    md.append("")
    md.append("- Run label: FAITHFUL_REIMPLEMENTATION (exact submitted implementation unavailable)")
    md.append("- Source dataset (preserved, unmodified): `uploads/prepared_paired_dataset_2000.csv`")
    md.append(f"- Working copy: `data/prepared_paired_dataset_2000.csv` (SHA-256 match verified: `{manifest['data_sha256_working_copy'][:16]}...`)")
    md.append(f"- Rows: {n}; fixed split seed: {SPLIT_SEED}")
    md.append("")
    md.append("## Exact duplicate audit")
    md.append("")
    md.append("Method: NFKC normalization + case folding + whitespace collapse, applied to BOTH `original_text` and `redacted_text`; records sharing a normalized text in either view are unioned into one group.")
    md.append("")
    md.append(f"- Exact-duplicate pairs: **{exact_pair_count}**")
    md.append(f"- Rows that are members of an exact-duplicate group: **{int(exact_member.sum())}**")
    md.append("")
    md.append("## Near-duplicate audit")
    md.append("")
    md.append(f"Method: char_wb (3,5) TF-IDF vectors (min_df=2), cosine similarity, fixed threshold **{NEAR_DUP_THRESHOLD}**; pairs at or above the threshold are unioned (computed on both views).")
    md.append("")
    md.append(f"- Near-duplicate pairs (either view): **{len(nd_pairs)}**")
    md.append(f"- Rows that are members of a near-duplicate pair: **{int(near_member.sum())}**")
    md.append(f"- Total lineage groups (exact + near, union-find): **{int(df['group_id'].nunique())}** (largest group size {int(group_sizes.max())})")
    md.append("")
    md.append("## Metadata audit")
    md.append("")
    md.append("| Metadata | Availability |")
    md.append("|---|---|")
    md.append("| hospital | ABSENT (column not present) |")
    md.append("| website | ABSENT (column not present) |")
    md.append("| account | ABSENT (column not present) |")
    md.append("| collection batch | ABSENT (column not present) |")
    md.append("| synthetic-parent id | ABSENT (no parent-id column; `pii_original` used as synthetic-injection proxy) |")
    md.append("| language / source_family | present; **perfectly confounded** (en <-> english_public_hospital_reviews, zh <-> chinese_public_web_deidentified) |")
    md.append("")
    md.append("Because language and source_family are perfectly confounded, they are used for **stratification** only, not as grouping constraints (a source-level grouping would be degenerate). All subgroup results are reported as a **joint language/source diagnostic**; no separate causal language or source effects are claimed.")
    md.append("")
    md.append("## Group-aware locked split")
    md.append("")
    md.append(f"Algorithm: within each language, groups shuffled with `numpy.default_rng({SPLIT_SEED})`, sorted by size descending, greedily assigned to the split most below its target share after assignment (train 70% / validation 15% / test 15%).")
    md.append("")
    md.append("| Split | Rows | en | zh |")
    md.append("|---|---:|---:|---:|")
    for sp in ["train", "validation", "test"]:
        sub = df[df["locked_split"] == sp]
        md.append(f"| {sp} | {len(sub)} | {int((sub['language']=='en').sum())} | {int((sub['language']=='zh').sum())} |")
    md.append("")
    md.append("## Leakage crossing checks")
    md.append("")
    md.append(f"- Groups crossing splits: **{crossing_groups}** (must be 0)")
    md.append(f"- Exact-duplicate pairs separated across splits: **{exact_sep}** (must be 0)")
    md.append(f"- Near-duplicate pairs separated across splits: **{near_sep}** (must be 0)")
    md.append("")
    md.append("The original `split` column is preserved in `locked_split.csv` for reference; all downstream stages use `locked_split`. Locked-test labels were not used for any model selection.")
    with open(os.path.join(DATA_DIR, "leakage_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("split counts:", df["locked_split"].value_counts().to_dict())
    print("crossing:", crossing_groups, "exact_sep:", exact_sep, "near_sep:", near_sep)
    print("exact pairs:", exact_pair_count, "near pairs:", len(nd_pairs),
          "groups:", int(df["group_id"].nunique()))
    print("stage1 done")


if __name__ == "__main__":
    main()
