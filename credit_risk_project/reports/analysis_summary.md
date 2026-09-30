# Analysis Summary

Cleaned loans: **1,344,949**
Issue-year coverage: **2007–2018**

| Experiment | Model | PR-AUC | ROC-AUC | Recall | F1 | Top-20% Capture | Lift |
|---|---|---:|---:|---:|---:|---:|---:|
| base | XGBoost | 0.3665 | 0.7007 | 0.8738 | 0.4143 | 36.9% | 1.85× |
| vietnam | XGBoost | 0.3665 | 0.6995 | 0.8517 | 0.4178 | 36.8% | 1.84× |

## How to interpret the business metric

Top-20% Capture answers the capacity-constrained review question: what share of defaults is found when reviewers focus on the 20% highest-risk applications?

Lift compares that capture concentration with random selection. A lift above 1 means the model concentrates defaults better than random review.

## Generated artifacts

- `eda_*.png` and grouped CSV tables
- `precision_recall_*.png`
- `gains_curve_*.png`
- `threshold_analysis_*.csv`
- `test_predictions_*.csv`
