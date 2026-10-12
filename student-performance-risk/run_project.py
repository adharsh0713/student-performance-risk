"""Command-line entry point for the student-risk prediction pipeline."""
from __future__ import annotations

import json
import sys
import time

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from project_config import (
    ARTIFACT_DIR, FIGURE_DIR, FEATURES, OUTPUT_DIR, PROCESSED_DIR,
    RANDOM_STATE, RAW_DIR, REPORT_DIR, WINDOW_DAYS, log,
)
from data_pipeline import build_features, get_dataset_paths
from eda import make_eda, write_eda_report
from modeling import make_model_pipelines, metric_row, run_shap, save_eval_artifacts
from reports import write_model_report

def main() -> None:
    start = time.time()
    for directory in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, REPORT_DIR, FIGURE_DIR, ARTIFACT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    paths = get_dataset_paths()
    data = build_features(paths)
    summary = make_eda(data)
    write_eda_report(summary)

    log("Preparing stratified train/test split.")
    X = data[FEATURES].copy()
    y = data["at_risk"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    model_pipelines = make_model_pipelines()
    results: list[dict] = []
    log("Training four baseline models.")
    for name, pipeline in model_pipelines.items():
        log(f"Training {name}.")
        pipeline.fit(X_train, y_train)
        metrics, _, _ = metric_row(name, pipeline, X_test, y_test)
        results.append(metrics)
        log(f"{name}: F1={metrics['f1_at_risk']:.3f}, recall={metrics['recall_at_risk']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")

    log("Tuning XGBoost using stratified 3-fold CV on the training set.")
    xgb_pipeline = model_pipelines["XGBoost"]
    parameter_distributions = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=parameter_distributions,
        n_iter=6,
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_metrics, tuned_pred, _ = metric_row("XGBoost (tuned)", tuned_model, X_test, y_test)
    results.append(tuned_metrics)
    log(f"Tuned XGBoost: F1={tuned_metrics['f1_at_risk']:.3f}, recall={tuned_metrics['recall_at_risk']:.3f}")

    save_eval_artifacts(results, y_test, tuned_pred)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    keep_cols = [c for c in ["rank_test_score", "mean_test_score", "std_test_score", "params"] if c in cv_results]
    cv_results[keep_cols].to_csv(REPORT_DIR / "tuning_results.csv", index=False)
    (REPORT_DIR / "classification_report.json").write_text(
        json.dumps(classification_report(y_test, tuned_pred, target_names=["not_at_risk", "at_risk"], output_dict=True, zero_division=0), indent=2),
        encoding="utf-8",
    )
    shap_note = run_shap(tuned_model, X_test)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, ARTIFACT_DIR / "student_risk_model.joblib")
    metadata = {
        "target": "at_risk",
        "target_definition": {"Fail": 1, "Withdrawn": 1, "Pass": 0, "Distinction": 0},
        "early_window_days": WINDOW_DAYS,
        "features": FEATURES,
        "positive_class": "Fail or Withdrawn",
        "model_type": "Tuned XGBoost pipeline with preprocessing",
        "random_state": RANDOM_STATE,
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_model_report(results, search, tuned_metrics, shap_note, summary)

    log("Completed successfully.")
    log(f"EDA report: {REPORT_DIR / 'eda_report.md'}")
    log(f"Model report: {REPORT_DIR / 'model_evaluation_report.md'}")
    log(f"Final model: {ARTIFACT_DIR / 'student_risk_model.joblib'}")
    log(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
