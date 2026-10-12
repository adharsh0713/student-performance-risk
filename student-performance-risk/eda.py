"""Exploratory data analysis and preprocessing report generation."""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_config import (
    FEATURES, FIGURE_DIR, NUMERIC_FEATURES, REPORT_DIR, WINDOW_DAYS,
)

def make_eda(data: pd.DataFrame) -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "rows": int(len(data)),
        "modules": int(data["code_module"].nunique()),
        "presentations": int(data["code_presentation"].nunique()),
        "at_risk_count": int(data["at_risk"].sum()),
        "at_risk_pct": float(data["at_risk"].mean() * 100),
        "window_days": WINDOW_DAYS,
        "final_result_counts": data["final_result"].value_counts().to_dict(),
        "missing_feature_pct": (data[FEATURES].isna().mean().mul(100).sort_values(ascending=False).round(2).to_dict()),
        "mean_clicks_at_risk": float(data.loc[data.at_risk == 1, "early_total_clicks"].mean()),
        "mean_clicks_not_at_risk": float(data.loc[data.at_risk == 0, "early_total_clicks"].mean()),
        "mean_early_score_at_risk": float(data.loc[data.at_risk == 1, "early_assessment_mean_score"].mean()),
        "mean_early_score_not_at_risk": float(data.loc[data.at_risk == 0, "early_assessment_mean_score"].mean()),
    }

    counts = data["final_result"].value_counts().reindex(["Pass", "Distinction", "Fail", "Withdrawn"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Distribution of Final Course Outcomes")
    ax.set_xlabel("Final result")
    ax.set_ylabel("Enrollment count")
    ax.bar_label(bars, fmt="%d")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    click_series = [
        np.log1p(data.loc[data.at_risk == 0, "early_total_clicks"].dropna()),
        np.log1p(data.loc[data.at_risk == 1, "early_total_clicks"].dropna()),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        click_series,
        tick_labels=["Not at risk", "At risk"],
        showfliers=False
    )
    ax.set_title("Early VLE Engagement by Risk Group")
    ax.set_ylabel(f"log(1 + clicks in first {WINDOW_DAYS} days)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "engagement_by_risk.png", dpi=160)
    plt.close(fig)

    risk_by_module = data.groupby("code_module", observed=True)["at_risk"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(risk_by_module.index.astype(str), risk_by_module.values * 100)
    ax.set_title("At-risk Rate by Course Module")
    ax.set_xlabel("Module")
    ax.set_ylabel("At-risk enrollment (%)")
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_by_module.png", dpi=160)
    plt.close(fig)

    numeric = data[NUMERIC_FEATURES + ["at_risk"]].select_dtypes(include=np.number)
    corr = numeric.corr(numeric_only=True)["at_risk"].drop("at_risk").dropna().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4776a8" if value < 0 else "#bf6b54" for value in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Numeric Feature Correlation with At-risk Label")
    ax.set_xlabel("Pearson correlation (descriptive, not causal)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_correlations.png", dpi=160)
    plt.close(fig)

    result_table = pd.DataFrame({"count": counts, "percentage": counts / len(data) * 100})
    result_table.to_csv(REPORT_DIR / "outcome_distribution.csv")
    missing_table = pd.Series(summary["missing_feature_pct"], name="missing_percent").to_frame()
    missing_table.to_csv(REPORT_DIR / "feature_missingness.csv")
    (REPORT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def write_eda_report(summary: dict) -> None:
    counts = summary["final_result_counts"]
    result_lines = "\n".join(
        f"| {result} | {int(counts.get(result, 0)):,} | {counts.get(result, 0) / summary['rows'] * 100:.2f}% |"
        for result in ["Pass", "Distinction", "Fail", "Withdrawn"]
    )
    missing_rows = sorted(summary["missing_feature_pct"].items(), key=lambda item: item[1], reverse=True)[:8]
    missing_lines = "\n".join(f"| `{name}` | {value:.2f}% |" for name, value in missing_rows)
    risk_clicks = summary["mean_clicks_at_risk"]
    safe_clicks = summary["mean_clicks_not_at_risk"]
    click_comparison = "higher" if safe_clicks > risk_clicks else "lower" if safe_clicks < risk_clicks else "similar"
    score_a = summary["mean_early_score_at_risk"]
    score_n = summary["mean_early_score_not_at_risk"]
    score_comparison = "higher" if score_n > score_a else "lower" if score_n < score_a else "similar"
    report = f"""# Student Performance Prediction — Data Preprocessing & EDA Report

**Dataset:** Open University Learning Analytics Dataset (OULAD)  
**Prediction objective:** Identify enrollments at risk of low final academic performance using information available in the first {WINDOW_DAYS} days.  
**Positive class:** `at_risk = 1` for `Fail` or `Withdrawn`; `at_risk = 0` for `Pass` or `Distinction`.

## 1. Project objective

Build an early-warning classifier that compares Logistic Regression, Decision Tree, Random Forest and XGBoost. Use SHAP to interpret the strongest predictors. The accompanying review paper describes OULAD as a commonly used learning-analytics dataset and highlights preprocessing, model comparison and explainability as important elements of student-performance prediction.

## 2. Dataset overview

The constructed modeling table contains **{summary['rows']:,} enrollment records**, covering **{summary['modules']} modules** and **{summary['presentations']} course presentations**. OULAD is relational: the pipeline joins student outcomes/demographics, registration, course metadata, assessment dates/results, and VLE interactions using module, presentation and student/assessment identifiers.

| Final outcome | Count | Percentage |
|---|---:|---:|
{result_lines}

The combined at-risk share is **{summary['at_risk_pct']:.2f}%** ({summary['at_risk_count']:,} enrollments). Metrics therefore report precision, recall and F1 in addition to accuracy; recall is especially relevant because missed at-risk students may not receive support.

## 3. Preprocessing and feature engineering

1. Read the seven OULAD CSV tables and keep valid final outcomes.
2. Join registration date and course presentation length; exclude `date_unregistration` because it can reveal the withdrawal outcome directly.
3. Aggregate non-exam assessment scores whose published due date and student submission date fall within days 0–{WINDOW_DAYS}.
4. Read the large `studentVle.csv` table in chunks and aggregate first-window total clicks, active days, distinct resources and recorded interaction rows.
5. Derive the target label, missed early assessments, and missingness/summary tables. Imputation and categorical encoding are fitted inside each model pipeline using the training set only.
6. Exclude student ID and final outcome columns from the model inputs. Use a stratified train/test split and a fixed random seed.

**Dataset limitation:** OULAD does not contain a direct classroom-attendance field. VLE activity (`early_active_days`, clicks and resources used) is used as an online-engagement/participation proxy, not as literal attendance. Features use only the first {WINDOW_DAYS} course days; this is an early-warning modeling choice, not a claim that the prediction is suitable for every course or institution.

## 4. Exploratory findings

- Mean clicks during the early window: **{safe_clicks:,.1f}** for non-at-risk enrollments and **{risk_clicks:,.1f}** for at-risk enrollments. The non-at-risk group has {click_comparison} average click activity in this descriptive summary.
- Mean early assessment score where available: **{score_n:.1f}** for non-at-risk enrollments and **{score_a:.1f}** for at-risk enrollments. The non-at-risk group has {score_comparison} average score where early scores are recorded. This comparison is descriptive and does not establish causation.
- The module-level risk rate and numeric correlations are included as exploratory plots. Results can vary across course module and presentation; aggregate patterns should not be interpreted as causal effects.

### Figures

![Final outcomes](figures/target_distribution.png)

![Early engagement by risk group](figures/engagement_by_risk.png)

![Risk rate by module](figures/risk_by_module.png)

![Numeric correlations](figures/numeric_correlations.png)

## 5. Missing values in model features

The model pipeline imputes numeric features with the training-set median and categorical features with the training-set most-frequent value. Top feature missingness before model imputation:

| Feature | Missing values (%) |
|---|---:|
{missing_lines}

## 6. Next stages included in the project package

- Baseline comparison of Logistic Regression, Decision Tree, Random Forest and XGBoost.
- Randomized cross-validated hyperparameter tuning of XGBoost on training data.
- Test-set evaluation, confusion matrix and classification report.
- SHAP feature-importance analysis (with a model-feature-importance fallback if SHAP cannot initialize).
- Saved fitted pipeline and a command-line script for predicting risk from a CSV of engineered features.

## 7. Responsible-use notes

These are predictions from anonymized historical online-course data, not determinations of student ability. Use outputs only to offer additional support, validate performance across subgroups, monitor false negatives/positives, and avoid punitive decisions. Patterns from the Open University context may not generalize to other institutions.

## Dataset citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics Dataset (OULAD).* Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

## Research review used as project background

Boujmiraz, S., Darhmaoui, H., & Drissi El Maliani, A. (2026). *Predicting student performance: A comprehensive review of machine learning, deep learning, and explainable AI approaches.* Computers and Education: Artificial Intelligence, 10, 100548. https://doi.org/10.1016/j.caeai.2026.100548
"""
    (REPORT_DIR / "eda_report.md").write_text(report, encoding="utf-8")
