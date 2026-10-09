"""Stage 1b: write data/leakage_audit.json (machine-readable leakage audit).

Generated deterministically from the already-saved stage-1 outputs
(data/leakage_audit.csv and data/locked_split_manifest.json); no new
measurements are invented here.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, RUN_LABEL


def main():
    audit = pd.read_csv(os.path.join(DATA_DIR, "leakage_audit.csv"))
    manifest = json.load(open(os.path.join(DATA_DIR, "locked_split_manifest.json")))

    group_sizes = audit.groupby("group_id")["group_size"].first()
    doc = {
        "run_label": RUN_LABEL,
        "created": "2026-10-09",
        "source_dataset_original": manifest["source_dataset_original"],
        "source_dataset_working_copy": manifest["source_dataset_working_copy"],
        "data_sha256_original": manifest["data_sha256_original"],
        "data_sha256_working_copy": manifest["data_sha256_working_copy"],
        "rows": int(manifest["rows"]),
        "split_seed": manifest["split_seed"],
        "split_targets": manifest["split_targets"],
        "split_counts": manifest["split_counts"],
        "language_counts_per_split": manifest["language_counts_per_split"],
        "source_family_counts_per_split": manifest["source_family_counts_per_split"],
        "language_source_confounded": manifest["language_source_confounded"],
        "duplicate_audit": manifest["duplicate_audit"],
        "group_stats": {
            "total_groups": int(manifest["duplicate_audit"]["total_groups"]),
            "singleton_groups": int((group_sizes == 1).sum()),
            "multi_record_groups": int((group_sizes > 1).sum()),
            "largest_group_size": int(group_sizes.max()),
            "rows_in_multi_record_groups": int(group_sizes[group_sizes > 1].sum()),
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
            "split_algorithm": "within each language, groups shuffled with numpy.default_rng(split_seed), "
                               "sorted by size descending, greedily assigned to the split most below "
                               "its target share after assignment (train 70% / validation 15% / test 15%)",
        },
        "per_record_detail": "data/leakage_audit.csv (one row per record: group membership, partner counts, flags, normalized-text hashes)",
        "split_manifest": "data/locked_split_manifest.json",
    }
    with open(os.path.join(DATA_DIR, "leakage_audit.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print("wrote data/leakage_audit.json")
    print(json.dumps({k: doc[k] for k in ["rows", "split_counts", "group_stats", "flag_counts",
                                          "crossing_counts"]}, indent=1))


if __name__ == "__main__":
    main()
