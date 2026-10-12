"""Dataset acquisition, cleaning, joins, and feature engineering."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import (
    KEYS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PROCESSED_DIR,
    RAW_DIR, REQUIRED_FILES, DOWNLOAD_URLS, WINDOW_DAYS, log,
)

def find_csv(filename: str) -> Path | None:
    matches = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.name.lower() == filename.lower()]
    return matches[0] if matches else None

def get_dataset_paths() -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: find_csv(name) for name in REQUIRED_FILES}
    if all(paths.values()):
        log("Found all seven OULAD CSV files locally.")
        return paths  # type: ignore[return-value]

    archive = RAW_DIR / "anonymisedData.zip"
    errors: list[str] = []
    for url in DOWNLOAD_URLS:
        try:
            log(f"Downloading dataset from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 student-performance-risk-project"})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if not zipfile.is_zipfile(archive):
                raise ValueError("The download was not a ZIP file (the source may have returned an error page).")
            with zipfile.ZipFile(archive) as zf:
                bad_member = zf.testzip()
                if bad_member:
                    raise ValueError(f"Corrupt ZIP member: {bad_member}")
                zf.extractall(RAW_DIR)
            paths = {name: find_csv(name) for name in REQUIRED_FILES}
            missing = [name for name, path in paths.items() if path is None]
            if not missing:
                log("Dataset downloaded and extracted.")
                return paths  # type: ignore[return-value]
            errors.append(f"{url}: missing files after extraction: {', '.join(missing)}")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not obtain all seven OULAD CSV files automatically.\n"
        "Download anonymisedData.zip manually from https://analyse.kmi.open.ac.uk/open_dataset "
        "or https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset, "
        f"extract the seven CSVs into {RAW_DIR}, and rerun.\nDetails: " + " | ".join(errors)
    )

def _read(paths: dict[str, Path], name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(paths[name], low_memory=False, **kwargs)

def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def build_features(paths: dict[str, Path]) -> pd.DataFrame:
    log("Loading enrollment-level student records.")
    students = _read(paths, "studentInfo.csv")
    students = students[students["final_result"].isin(["Pass", "Distinction", "Fail", "Withdrawn"])].copy()
    students = students.drop_duplicates(KEYS)

    registration = _read(paths, "studentRegistration.csv", usecols=KEYS + ["date_registration"])
    registration = registration.drop_duplicates(KEYS)
    registration["date_registration"] = _numeric(registration["date_registration"])
    data = students.merge(registration, on=KEYS, how="left", validate="one_to_one")

    courses = _read(paths, "courses.csv", usecols=["code_module", "code_presentation", "module_presentation_length"])
    data = data.merge(
        courses.drop_duplicates(["code_module", "code_presentation"]),
        on=["code_module", "code_presentation"], how="left",
    ).rename(columns={"module_presentation_length": "course_length"})

    # Early assessment features: only non-exam assessments due and submitted within the first N days.
    log(f"Building assessment features from days 0–{WINDOW_DAYS}.")
    assessments = _read(paths, "assessments.csv")
    assessments["date"] = _numeric(assessments["date"])
    early_assessments = assessments[
        assessments["date"].between(0, WINDOW_DAYS) & assessments["assessment_type"].ne("Exam")
    ].copy()
    due = (early_assessments.groupby(["code_module", "code_presentation"])["id_assessment"]
           .nunique().rename("early_assessments_due").reset_index())
    data = data.merge(due, on=["code_module", "code_presentation"], how="left")

    submissions = _read(paths, "studentAssessment.csv")
    submissions["date_submitted"] = _numeric(submissions["date_submitted"])
    submissions["score"] = _numeric(submissions["score"])
    eligible = submissions.merge(
        early_assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment", how="inner", validate="many_to_one",
    )
    eligible = eligible[
        eligible["date_submitted"].between(0, WINDOW_DAYS)
        & eligible["score"].notna()
    ].copy()
    if len(eligible):
        assessment_features = eligible.groupby(KEYS).agg(
            early_assessment_mean_score=("score", "mean"),
            early_assessment_min_score=("score", "min"),
            early_assessment_score_std=("score", lambda s: s.std(ddof=0)),
            early_assessments_submitted=("id_assessment", "nunique"),
            early_low_scores=("score", lambda s: int((s < 40).sum())),
        ).reset_index()
        data = data.merge(assessment_features, on=KEYS, how="left")
    else:
        for col in ["early_assessment_mean_score", "early_assessment_min_score", "early_assessment_score_std"]:
            data[col] = np.nan
        data["early_assessments_submitted"] = 0
        data["early_low_scores"] = 0

    # Chunk the ~10M-row clickstream; keep only the early window before aggregation.
    log(f"Aggregating VLE interactions from days 0–{WINDOW_DAYS} (chunked read).")
    early_chunks: list[pd.DataFrame] = []
    usecols = KEYS + ["id_site", "date", "sum_click"]
    for chunk in pd.read_csv(paths["studentVle.csv"], usecols=usecols, chunksize=750_000, low_memory=False):
        chunk["date"] = _numeric(chunk["date"])
        chunk["sum_click"] = _numeric(chunk["sum_click"]).fillna(0)
        selected = chunk[chunk["date"].between(0, WINDOW_DAYS)]
        if not selected.empty:
            early_chunks.append(selected)
    if early_chunks:
        vle_early = pd.concat(early_chunks, ignore_index=True)
        vle_features = vle_early.groupby(KEYS).agg(
            early_total_clicks=("sum_click", "sum"),
            early_active_days=("date", "nunique"),
            early_resources_used=("id_site", "nunique"),
            early_vle_records=("date", "size"),
        ).reset_index()
        data = data.merge(vle_features, on=KEYS, how="left")
    else:
        for col in ["early_total_clicks", "early_active_days", "early_resources_used", "early_vle_records"]:
            data[col] = 0

    for col in ["early_assessments_due", "early_assessments_submitted", "early_low_scores",
                "early_total_clicks", "early_active_days", "early_resources_used", "early_vle_records"]:
        if col not in data:
            data[col] = 0
        data[col] = _numeric(data[col]).fillna(0)
    data["early_missed_assessments"] = (data["early_assessments_due"] - data["early_assessments_submitted"]).clip(lower=0)
    data["at_risk"] = data["final_result"].map({"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1})
    data = data[data["at_risk"].notna()].copy()
    data["at_risk"] = data["at_risk"].astype(int)

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = data[col].replace("?", np.nan)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DIR / "student_features.csv", index=False)
    log(f"Saved feature table: {len(data):,} enrollments × {data.shape[1]} columns.")
    return data
