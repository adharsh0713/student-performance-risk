# Faculty deliverables checklist

| Faculty deliverable | Included implementation | File(s) generated after `python run_project.py` |
|---|---|---|
| Data preprocessing and exploratory analysis report | Seven-table joins, cleaning, first-30-day features, target analysis and four EDA figures | `reports/eda_report.md`, `reports/figures/` |
| Baseline model implementation | Logistic Regression, Decision Tree, Random Forest and XGBoost | `reports/model_comparison.csv` |
| Advanced model integration, tuning and evaluation | RandomizedSearchCV over XGBoost with stratified 3-fold CV; holdout metrics and confusion matrix | `reports/model_evaluation_report.md`, `reports/tuning_results.csv`, `reports/classification_report.json` |
| Model optimization and deployment-ready script/notebook | Saved preprocessing + tuned XGBoost pipeline, metadata and CSV batch prediction | `artifacts/student_risk_model.joblib`, `artifacts/model_metadata.json`, `predict_risk.py`, `Student_Performance_Prediction.ipynb` |

**Before submitting:** run the pipeline against the real OULAD dataset and use the generated reports/metrics. Synthetic smoke-test outputs are not project results and are not included in this package. Cite both the OULAD dataset and the faculty-provided review paper. Change the 30-day feature window only if the faculty gives another prediction horizon.
