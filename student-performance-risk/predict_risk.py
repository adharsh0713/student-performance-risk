"""Batch inference using the fitted OULAD student-risk model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "student_risk_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict at-risk probability for pre-engineered student enrollment features.")
    parser.add_argument("--input", required=True, help="CSV containing engineered feature columns from the training pipeline.")
    parser.add_argument("--output", default="predictions.csv", help="Where to write predictions (default: predictions.csv).")
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model artifact is missing. Run `python run_project.py` first.")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = metadata["features"]
    data = pd.read_csv(args.input)
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("Input CSV is missing engineered feature columns: " + ", ".join(missing))
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(data[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result = data.copy()
    result["predicted_at_risk"] = predictions
    result["at_risk_probability"] = probabilities.round(5)
    result["risk_label"] = result["predicted_at_risk"].map({1: "At risk: supportive review recommended", 0: "Not flagged by model"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} predictions to {output.resolve()}")


if __name__ == "__main__":
    main()
