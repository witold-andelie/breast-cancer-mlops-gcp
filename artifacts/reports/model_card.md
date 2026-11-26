# Model Card: `random_forest` (v1)

## 1. Model Overview & Lineage
- **Stage**: `production`
- **Run ID**: `a9f897952af7`
- **Created At (UTC)**: `2026-11-26T20:10:56.865977+00:00`
- **Dataset SHA-256**: `aa47becd63cb0ef6f9fd7457bbc67be1bb0d58da61fd341dd0981cbbe75dc2c9`
- **Deduplication**: `deduplicate_features=True` (raw=1138 rows -> clean=569 unique patients, removed=569 duplicate twins)
- **Partition**: Train = 418 rows, Test = 151 rows (deterministic DuckDB hash split)
- **Clinical Decision Threshold**: `0.32` (tuned on CV out-of-fold probabilities to prioritize malignant Recall_M)

## 2. Quality Gate Verification
- PASS recall_M=0.9825 >= 0.9500
- PASS roc_auc=0.9853 >= 0.9800
- PASS brier_score=0.0427 <= 0.0600
- PASS probabilities finite and bounded in [0, 1]

## 3. Held-Out Test Performance (`cv_tuned` thresholds)
| model | eval_mode | threshold | accuracy | recall_M | precision_M | f1 | f2 | roc_auc | pr_auc | brier_score | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_l2 | cv_tuned | 0.22 | 0.9536 | 0.9649 | 0.9167 | 0.9402 | 0.9549 | 0.9961 | 0.9942 | 0.0225 | 55 | 5 | 89 | 2 |
| logistic_statsmodels | cv_tuned | 0.17 | 0.9536 | 0.9825 | 0.9032 | 0.9412 | 0.9655 | 0.9961 | 0.9942 | 0.0235 | 56 | 6 | 88 | 1 |
| random_forest | cv_tuned | 0.32 | 0.9338 | 0.9825 | 0.8615 | 0.918 | 0.9556 | 0.9853 | 0.9815 | 0.0427 | 56 | 9 | 85 | 1 |
| hist_gbdt | cv_tuned | 0.13 | 0.9603 | 0.9825 | 0.918 | 0.9492 | 0.9689 | 0.9965 | 0.9947 | 0.0182 | 56 | 5 | 89 | 1 |
| chaid_bonferroni | cv_tuned | 0.13 | 0.6689 | 1.0 | 0.5327 | 0.6951 | 0.8507 | 0.9455 | 0.9029 | 0.0753 | 57 | 50 | 44 | 0 |
| chaid_unadjusted | cv_tuned | 0.13 | 0.6689 | 1.0 | 0.5327 | 0.6951 | 0.8507 | 0.9455 | 0.9029 | 0.0753 | 57 | 50 | 44 | 0 |

## 4. 5-Fold Cross-Validation Summary (`cv_tuned` thresholds)
| model | eval_mode | threshold | accuracy | recall_M | precision_M | f1 | f2 | roc_auc | pr_auc | brier_score | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_l2 | cv_tuned | 0.22 | 0.9426 | 0.9613 | 0.8922 | 0.9255 | 0.9466 | 0.9877 | 0.9851 | 0.0337 | 149 | 18 | 245 | 6 |
| logistic_statsmodels | cv_tuned | 0.17 | 0.9402 | 0.9613 | 0.8869 | 0.9226 | 0.9454 | 0.9874 | 0.9848 | 0.0346 | 149 | 19 | 244 | 6 |
| random_forest | cv_tuned | 0.32 | 0.9402 | 0.9806 | 0.8736 | 0.924 | 0.9572 | 0.9911 | 0.9878 | 0.0345 | 152 | 22 | 241 | 3 |
| hist_gbdt | cv_tuned | 0.13 | 0.9593 | 0.9677 | 0.9259 | 0.9464 | 0.9591 | 0.9865 | 0.9857 | 0.0248 | 150 | 12 | 251 | 5 |
| chaid_bonferroni | cv_tuned | 0.13 | 0.7177 | 0.9871 | 0.5688 | 0.7217 | 0.8605 | 0.9409 | 0.8919 | 0.0631 | 153 | 116 | 147 | 2 |
| chaid_unadjusted | cv_tuned | 0.13 | 0.7703 | 0.9742 | 0.6214 | 0.7588 | 0.8749 | 0.9458 | 0.9071 | 0.0737 | 151 | 92 | 171 | 4 |

## 5. Clinical Subgroup Slice Evaluation (`radius_worst` tertiles)
| slice | n | malignant_cnt | accuracy | recall_M | precision_M | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| radius_worst=low | 44 | 0 | 0.9773 | 0.0 | 0.0 | 0 | 1 | 43 | 0 |
| radius_worst=mid | 56 | 10 | 0.9107 | 0.9 | 0.6923 | 9 | 4 | 42 | 1 |
| radius_worst=high | 51 | 47 | 0.9216 | 1.0 | 0.9216 | 47 | 4 | 0 | 0 |

## 6. Unsupervised Structure & Anomaly Insights (Train-Fitted)
- **PCA**: 7 components explain 90% variance; 10 components explain 95% variance.
- **K-Means (k=2, unlabeled)**: Train ARI=0.6824, Test ARI=0.6376; Best silhouette k=2 (0.3415).
- **Isolation Forest (5% contamination)**: Malignancy rate inside flagged train anomalies is 81.0% vs 34.8% in normal samples.

## 7. Intended Use, Limitations & Ethical Notes
- **Intended Use**: Decision-support triage prototype for breast mass malignancy scoring; positive (`M`) misses carry high clinical cost, so recall is prioritized over accuracy.
- **Data Hygiene & Leakage Guard**: Raw CSV contains 1138 rows across 569 unique feature vectors (`A`/`B` paired ID prefixes). Pre-split SQL deduplication removes exact duplicate twins before partitioning so zero patient feature signatures leak between train and test.
- **Limitations**: Trained on static diagnostic measurements; production use requires continuous PSI/KS drift checks and clinician oversight.