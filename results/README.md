# Numerical results

| Directory/file | Contents |
|---|---|
| `official/canonical_per_run.csv` | Fifteen official-protocol runs with image/pixel ranking, overlap and thresholds. Private artifact-location columns are omitted; numerical cells are unchanged. |
| `official/localization_per_seed.csv` | Historical nine-run localization/boundary table at its recorded six-decimal precision. |
| `official/p01_supervision.csv` | Mixed, Real-only and Synthetic-only source controls, including image misses/false alarms and original-scale localization. |
| `official/p01_differences.csv` | Same-seed source differences relative to Mixed. |
| `diagnostics/` | Saved-prediction rechecks and threshold analyses. Matched/test-oracle points use test labels and are diagnostic, not formal gains. |
| `grouped/per_seed_results.csv` | Six grouped model runs under 20-defect primary and one-defect diagnostic calibration (12 rows). |
| `grouped/summary_mean_sample_sd.csv` | Grouped means and sample SDs, separated by mode and calibration budget. |
| `grouped/paired_*.csv` | Mixed-minus-Real-only differences for matched seeds and their summaries. |
| `grouped/split_*.csv`, `label_budget_group_counts.csv`, `grouping_rules.json` | Pool/label budgets, deterministic split hashes and grouping criteria. |

## Reading the tables

Continuous summaries use the mean and sample standard deviation across seeds 42, 7 and 2024. Counts of zero-overlap failures sum runs; the same test image can contribute once per run. Boundary-distance means include only valid predicted/reference contours, so valid-case and failure counts must also be read.

The official tests contain 180 normal and 180 defective images. Grouped tests contain 120 normal and 120 defective images. The grouped candidate pools are 360/360, 120/120 and 120/120 normal/defective images; selected training and primary validation sets use 336/3 and 84/20. All paired runs use the same selected lists. AP and AUROC do not depend on threshold selection.

P10 AP and Dice retain their recorded evaluation; its spatial IoU comes from the later spatial evaluation. The localization CSV preserves rounded historical values. Original-scale values and histogram-versus-continuous threshold definitions can differ slightly at bin boundaries; use each table's named fields and provenance, not a newly mixed metric record.

The official pools are non-nested and have detected cross-split content associations. The grouped supplement isolates known content relations and retrains the compared conditions. These tables describe their own protocols; they do not establish production-line performance or generalization to other defect types.
