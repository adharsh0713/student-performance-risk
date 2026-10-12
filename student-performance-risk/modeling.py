"""Model preprocessing, training pipelines, metrics, and explainability."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from project_config import (
    CATEGORICAL_FEATURES, FEATURES, FIGURE_DIR, NUMERIC_FEATURES,
    RANDOM_STATE, REPORT_DIR,
)

def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

def metric_row(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_at_risk": precision_score(y_test, predictions, zero_division=0),
        "recall_at_risk": recall_score(y_test, predictions, zero_division=0),
        "f1_at_risk": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
    }
    return metrics, predictions, probabilities

def make_model_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", n_estimators=250,
                max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
    }

def save_eval_artifacts(results: list[dict], y_test: pd.Series, predictions: np.ndarray) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).sort_values(["f1_at_risk", "recall_at_risk"], ascending=False)
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(table))
    width = 0.25
    for i, metric in enumerate(["precision_at_risk", "recall_at_risk", "f1_at_risk"]):
        bars = ax.bar(x + (i - 1) * width, table[metric], width=width, label=metric.replace("_at_risk", "").title())
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Baseline and Tuned Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Tuned XGBoost — Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Not at risk", "At risk"])
    ax.set_yticks([0, 1], labels=["Not at risk", "At risk"])
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tuned_xgboost_confusion_matrix.png", dpi=160)
    plt.close(fig)

def run_shap(best_model: Pipeline, X_test: pd.DataFrame) -> str:
    model = best_model.named_steps["model"]
    preprocessor = best_model.named_steps["preprocessor"]
    sample = X_test.sample(n=min(400, len(X_test)), random_state=RANDOM_STATE)
    transformed = preprocessor.transform(sample)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        shap.summary_plot(values, transformed, feature_names=feature_names, max_display=20, show=False)
        plt.gcf().set_size_inches(10, 7)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close("all")
        importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        importance.sort_values("mean_abs_shap", ascending=False).to_csv(REPORT_DIR / "feature_importance.csv", index=False)
        return "SHAP summary generated; see reports/figures/shap_summary.png and reports/feature_importance.csv."
    except Exception as exc:
        # Preserve a useful explainability artifact if SHAP is incompatible with the installed environment.
        importance = pd.DataFrame({"feature": feature_names, "model_importance": model.feature_importances_})
        importance.sort_values("model_importance", ascending=False).to_csv(REPORT_DIR / "feature_importance.csv", index=False)
        top = importance.sort_values("model_importance", ascending=False).head(20).sort_values("model_importance")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["feature"], top["model_importance"])
        ax.set_title("XGBoost Feature Importance (SHAP fallback)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "shap_summary.png", dpi=160)
        plt.close(fig)
        return f"SHAP could not initialize ({type(exc).__name__}: {exc}); saved built-in XGBoost importance as a fallback."
