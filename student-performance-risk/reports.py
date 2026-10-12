"""Write the model training, tuning, and evaluation report."""
from __future__ import annotations

import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from project_config import RANDOM_STATE, REPORT_DIR, WINDOW_DAYS

def write_model_report(results: list[dict], search: RandomizedSearchCV, tuned_metrics: dict, shap_note: str, report_obj: dict) -> None:
    table = pd.DataFrame(results).sort_values("f1_at_risk", ascending=False)
    lines = "\n".join(
        f"| {row['model']} | {row['accuracy']:.3f} | {row['precision_at_risk']:.3f} | {row['recall_at_risk']:.3f} | {row['f1_at_risk']:.3f} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} |"
        for _, row in table.iterrows()
    )
    best = table.iloc[0]
    report = f"""# Model Training, Tuning & Evaluation Report

## Experimental setup

- Dataset: OULAD-derived enrollment table (`data/processed/student_features.csv`).
- Target: at-risk = Fail or Withdrawn; not-at-risk = Pass or Distinction.
- Features are derived from course days 0–{WINDOW_DAYS}; final outcomes, student IDs, and withdrawal dates are excluded from the model inputs.
- Stratified train/test split: 80% / 20%; random seed: {RANDOM_STATE}.
- Numeric values are median-imputed and standardized. Categorical values are mode-imputed and one-hot encoded. Preprocessing is fitted inside each training pipeline.
- Positive-class metrics concern the at-risk group.

## Baseline comparison and tuned XGBoost

| Model | Accuracy | Precision (at-risk) | Recall (at-risk) | F1 (at-risk) | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
{lines}

The strongest model in this test-set table by F1 is **{best['model']}** ({best['f1_at_risk']:.3f}). A high recall means fewer at-risk enrollments are missed; precision measures how many flagged enrollments really are at risk. Use the full metric table rather than accuracy alone, especially for any imbalanced outcome.

## Hyperparameter tuning

XGBoost was tuned with randomized search and stratified 3-fold cross-validation on the training split only, optimizing F1 for the at-risk class. Best mean cross-validation F1: **{search.best_score_:.3f}**.

Best parameters:

```json
{json.dumps(search.best_params_, indent=2)}
```

## Evaluation artifacts

- Baseline/tuned metric table: `reports/model_comparison.csv`
- Model comparison: `reports/figures/model_comparison.png`
- Confusion matrix: `reports/figures/tuned_xgboost_confusion_matrix.png`
- Feature importance: `reports/feature_importance.csv`
- Explainability: {shap_note}
- Serialized fitted pipeline: `artifacts/student_risk_model.joblib`
- Inference schema: `artifacts/model_metadata.json`

## Interpretation and limitations

Metrics reflect this split of the OULAD historical data and must not be reported as universal model accuracy. The first-30-day feature window improves alignment with early intervention but can disadvantage courses with different schedules. VLE clicks are an engagement proxy, not proof of learning. Fairness and subgroup error rates should be audited before operational use. Predictions should trigger supportive review rather than punitive decisions.
"""
    (REPORT_DIR / "model_evaluation_report.md").write_text(report, encoding="utf-8")
