"""Central configuration and shared constants for the student-risk project."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"

PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "outputs"

REPORT_DIR = ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

ARTIFACT_DIR = ROOT / "artifacts"

WINDOW_DAYS = int(os.getenv("EARLY_WINDOW_DAYS", "30"))

RANDOM_STATE = 42

KEYS = ["code_module", "code_presentation", "id_student"]

REQUIRED_FILES = [
    "courses.csv", "assessments.csv", "studentAssessment.csv", "studentInfo.csv",
    "studentRegistration.csv", "vle.csv", "studentVle.csv",
]

DOWNLOAD_URLS = [
    "https://figshare.com/ndownloader/files/8606371",
    "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip",
]

NUMERIC_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "date_registration", "course_length",
    "early_total_clicks", "early_active_days", "early_resources_used", "early_vle_records",
    "early_assessment_mean_score", "early_assessment_min_score", "early_assessment_score_std",
    "early_assessments_due", "early_assessments_submitted", "early_low_scores", "early_missed_assessments",
]

CATEGORICAL_FEATURES = [
    "code_module", "code_presentation", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def log(message: str) -> None:
    print(f"[project] {message}", flush=True)
