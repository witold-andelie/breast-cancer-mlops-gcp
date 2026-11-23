# Model Card: `hist_gbdt` (v1)

## 1. Model Overview & Lineage
- **Stage**: `production`
- **Run ID**: `f17723a5ee48`
- **Created At (UTC)**: `2026-11-23T16:03:30.952507+00:00`
- **Dataset SHA-256**: `aa47becd63cb0ef6f9fd7457bbc67be1bb0d58da61fd341dd0981cbbe75dc2c9`
- **Partition**: Train = 805 rows, Test = 333 rows (deterministic DuckDB hash split)
- **Clinical Decision Threshold**: `0.21` (tuned on CV out-of-fold probabilities to prioritize malignant Recall_M)

## 2. Quality Gate Verification
- PASS recall_M=1.0000 >= 0.9500
- PASS roc_auc=1.0000 >= 0.9800
- PASS brier_score=0.0012 <= 0.0600
- PASS probabilities finite and bounded in [0, 1]

## 3. Held-Out Test Performance (`cv_tuned` thresholds)
| model | eval_mode | threshold | accuracy | recall_M | precision_M | f1 | f2 | roc_auc | pr_auc | brier_score | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_l2 | cv_tuned | 0.29 | 0.967 | 0.9774 | 0.942 | 0.9594 | 0.9701 | 0.9971 | 0.9958 | 0.0217 | 130 | 8 | 192 | 3 |
| logistic_statsmodels | cv_tuned | 0.16 | 0.958 | 0.985 | 0.9161 | 0.9493 | 0.9704 | 0.9971 | 0.9958 | 0.0221 | 131 | 12 | 188 | 2 |
| random_forest | cv_tuned | 0.46 | 0.985 | 0.9925 | 0.9706 | 0.9814 | 0.988 | 0.9987 | 0.9983 | 0.0184 | 132 | 4 | 196 | 1 |
| hist_gbdt | cv_tuned | 0.21 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0012 | 133 | 0 | 200 | 0 |
| chaid_bonferroni | cv_tuned | 0.1 | 0.8709 | 1.0 | 0.7557 | 0.8608 | 0.9393 | 0.976 | 0.9447 | 0.0549 | 133 | 43 | 157 | 0 |
| chaid_unadjusted | cv_tuned | 0.1 | 0.8018 | 1.0 | 0.6683 | 0.8012 | 0.9097 | 0.976 | 0.9447 | 0.0557 | 133 | 66 | 134 | 0 |

## 4. 5-Fold Cross-Validation Summary (`cv_tuned` thresholds)
| model | eval_mode | threshold | accuracy | recall_M | precision_M | f1 | f2 | roc_auc | pr_auc | brier_score | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_l2 | cv_tuned | 0.29 | 0.954 | 0.9725 | 0.9071 | 0.9386 | 0.9587 | 0.9939 | 0.9903 | 0.03 | 283 | 29 | 485 | 8 |
| logistic_statsmodels | cv_tuned | 0.16 | 0.9429 | 0.9828 | 0.8746 | 0.9256 | 0.9591 | 0.9939 | 0.9903 | 0.0307 | 286 | 41 | 473 | 5 |
| random_forest | cv_tuned | 0.46 | 0.9752 | 0.9759 | 0.9562 | 0.966 | 0.9719 | 0.9947 | 0.9938 | 0.0245 | 284 | 13 | 501 | 7 |
| hist_gbdt | cv_tuned | 0.21 | 0.9876 | 0.9897 | 0.9763 | 0.9829 | 0.987 | 0.9984 | 0.9975 | 0.0084 | 288 | 7 | 507 | 3 |
| chaid_bonferroni | cv_tuned | 0.1 | 0.7913 | 0.9622 | 0.6407 | 0.7692 | 0.8745 | 0.9577 | 0.9399 | 0.0725 | 280 | 157 | 357 | 11 |
| chaid_unadjusted | cv_tuned | 0.1 | 0.7913 | 0.9622 | 0.6407 | 0.7692 | 0.8745 | 0.9577 | 0.9399 | 0.0725 | 280 | 157 | 357 | 11 |

## 5. Clinical Subgroup Slice Evaluation (`radius_worst` tertiles)
| slice | n | malignant_cnt | accuracy | recall_M | precision_M | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| radius_worst=low | 101 | 1 | 1.0 | 1.0 | 1.0 | 1 | 0 | 100 | 0 |
| radius_worst=mid | 113 | 20 | 1.0 | 1.0 | 1.0 | 20 | 0 | 93 | 0 |
| radius_worst=high | 119 | 112 | 1.0 | 1.0 | 1.0 | 112 | 0 | 7 | 0 |

## 6. Unsupervised Structure & Anomaly Insights (Train-Fitted)
- **PCA**: 7 components explain 90% variance; 10 components explain 95% variance.
- **K-Means (k=2, unlabeled)**: Train ARI=0.6627, Test ARI=0.6507; Best silhouette k=2 (0.3488).
- **Isolation Forest (5% contamination)**: Malignancy rate inside flagged train anomalies is 75.6% vs 34.0% in normal samples.

## 7. Intended Use, Limitations & Ethical Notes
- **Intended Use**: Decision-support triage prototype for breast mass malignancy scoring; positive (`M`) misses carry high clinical cost, so recall is prioritized over accuracy.
- **Data Hygiene Note**: The source table has 1138 rows (569 unique feature vectors across paired IDs). Splitting on `id_pat` matches the course benchmark; real clinical deployment should deduplicate or group-split by patient/sample signature.
- **Limitations**: Trained on static diagnostic measurements; production use requires continuous PSI/KS drift checks and clinician oversight.