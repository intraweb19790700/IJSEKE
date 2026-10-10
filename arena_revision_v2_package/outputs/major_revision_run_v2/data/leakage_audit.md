# Leakage Audit and Group-Aware Locked Split

- Run label: `CORRECTED_FAITHFUL_REIMPLEMENTATION` (corrected faithful reimplementation; the exact submitted implementation is unavailable)
- Source dataset (preserved, unmodified): `/home/user/IJSEKE/arena_revision_v2_package/input/prepared_paired_dataset_2000.csv`
- Data SHA-256: `bf3ba79f69fc7b9f9b5c7497e592eec5b2f45b0483ee8db8aa8179797baed91e` (verified in stage 1)
- Rows: 2000; fixed split seed: 20260720

## Exact duplicate audit

Method: NFKC normalization + case folding + whitespace collapse, applied to BOTH `original_text` and `redacted_text`; records sharing a normalized text in either view are unioned into one group.

- Exact-duplicate pairs: **120**
- Rows that are members of an exact-duplicate group: **85**

## Near-duplicate audit

Method: char_wb (3,5) TF-IDF vectors (min_df=2), cosine similarity, fixed threshold **0.9**; pairs at or above the threshold are unioned (computed on both views).

- Near-duplicate pairs (either view): **448**
- Rows that are members of a near-duplicate pair: **383**
- Total lineage groups (exact + near, union-find): **1755** (largest group size 13)

## Metadata audit and limitations

| Metadata | Availability |
|---|---|
| hospital | ABSENT (column not present) |
| website | ABSENT (column not present) |
| account | ABSENT (column not present) |
| collection batch | ABSENT (column not present) |
| synthetic-parent id | ABSENT (no parent-id column; `pii_original` used as synthetic-injection proxy) |
| language / source_family | present; **perfectly confounded** (en <-> english_public_hospital_reviews, zh <-> chinese_public_web_deidentified) |

Because language and source_family are perfectly confounded, they are used for **stratification** only, not as grouping constraints (a source-level grouping would be degenerate: one group per language). All subgroup results are reported as a **joint language/source diagnostic**; no separate causal language or source effects are claimed.

## Group-aware locked split

Algorithm: within each language, groups shuffled with `numpy.default_rng(20260720)`, sorted by size descending, greedily assigned to the split most below its target share after assignment (train 70% / validation 15% / test 15%).

| Split | Rows | en | zh |
|---|---:|---:|---:|
| train | 1400 | 666 | 734 |
| validation | 301 | 143 | 158 |
| test | 299 | 142 | 157 |

## Leakage crossing checks

- Groups crossing splits: **0** (must be 0)
- Exact-duplicate pairs separated across splits: **0** (must be 0)
- Near-duplicate pairs separated across splits: **0** (must be 0)

The original `split` column is preserved in `locked_split.csv` for reference; all downstream stages use `locked_split`. Locked-test labels were not used for any model selection.
