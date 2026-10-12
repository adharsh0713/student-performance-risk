# Student Performance Prediction and Early Risk Detection

**Course project:** Predict students at risk of low final academic performance using early-course information; compare Logistic Regression, Decision Tree, Random Forest and XGBoost; tune XGBoost; explain important predictors with SHAP.

The project is based on the faculty-provided review paper *Predicting student performance: A comprehensive review of machine learning, deep learning, and explainable AI approaches* and uses OULAD, the Open University Learning Analytics Dataset.

## Dataset link (faculty's missing link)

- Open University OULAD page: https://analyse.kmi.open.ac.uk/open_dataset
- Figshare download page for `anonymisedData.zip` (44.58 MB): https://doi.org/10.6084/m9.figshare.5081998.v1
- UCI Machine Learning Repository mirror: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Research paper describing OULAD: https://doi.org/10.1038/sdata.2017.171

The paper's supplied access link is the old Open University address. If its download button fails, download `anonymisedData.zip` from the OULAD page or UCI page and extract the seven CSVs into `data/raw/`. The script also attempts an automatic download from the direct Figshare file URL first and UCI second.

## One-run execution

Install Python 3.10–3.12, then from this folder run:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_project.py
```

Ubuntu / macOS:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_project.py
```

The script downloads/extracts the dataset if needed, generates an early-window feature table and EDA report, trains all baseline models, tunes XGBoost with 3-fold randomized search, reports test metrics, produces SHAP (or a documented built-in feature-importance fallback), and saves the model. First run reads the large `studentVle.csv` file in chunks and may take several minutes.

### Google Colab

1. Download `Run_in_Colab.ipynb` separately from this project ZIP and upload that notebook to Google Colab.
2. Run all cells. The first cell asks you to upload this project ZIP, then it installs requirements and runs the pipeline.
3. When it finishes, download the generated reports, figures, processed feature table and model artifact from the final cell.

## Target and modeling choices

- `at_risk = 1`: final result is `Fail` or `Withdrawn`.
- `at_risk = 0`: final result is `Pass` or `Distinction`.
- Prediction window: first 30 course days by default. Change with `EARLY_WINDOW_DAYS=45` before running if your faculty requests another interval.
- VLE engagement variables: total clicks, active days, distinct resources and interaction records in that window.
- Assessment variables: mean/minimum score, score standard deviation, count of assessments due/submitted, low scores and missed early assessments.
- Additional predictors: module/presentation, prior attempts, studied credits, registration timing, course length and selected demographics.
- `date_unregistration`, final outcome and student ID are excluded from model inputs. Withdrawal dates would leak the target; student IDs are identifiers, not explanatory features.
- OULAD does not measure physical/classroom attendance directly. The project uses online VLE activity as an engagement/participation proxy, which must not be described as literal attendance.
- Train/test split is stratified (80/20). Imputation and encoding are inside the model pipelines. Metrics include accuracy, precision, recall, F1, ROC-AUC and PR-AUC.

## Deliverables produced automatically

| Deliverable | Output |
|---|---|
| Data preprocessing + EDA report | `reports/eda_report.md` |
| Clean engineered feature data | `data/processed/student_features.csv` |
| EDA tables and figures | `reports/outcome_distribution.csv`, `reports/feature_missingness.csv`, `reports/figures/` |
| Baseline model comparison | `reports/model_comparison.csv` |
| Hyperparameter search results | `reports/tuning_results.csv` |
| Evaluation report + classification report | `reports/model_evaluation_report.md`, `reports/classification_report.json` |
| SHAP/global importance | `reports/figures/shap_summary.png`, `reports/feature_importance.csv` |
| Deployment-ready serialized pipeline | `artifacts/student_risk_model.joblib` |
| Model/input metadata | `artifacts/model_metadata.json` |
| Batch prediction script | `predict_risk.py` |

## Batch prediction

After `run_project.py` completes, submit a CSV whose columns include the engineered features listed in `artifacts/model_metadata.json`:

```bash
python predict_risk.py --input path/to/engineered_students.csv --output predictions.csv
```

For a quick pipeline check only, the generated feature table can be used as the input (it contains existing labels, so do not present those rows as genuinely new students). For live deployment, upstream ingestion must compute the same first-30-day feature definitions for new enrollments.

## Research grounding and citation

Boujmiraz, S., Darhmaoui, H., & Drissi El Maliani, A. (2026). *Predicting student performance: A comprehensive review of machine learning, deep learning, and explainable AI approaches*. Computers and Education: Artificial Intelligence, 10, 100548. https://doi.org/10.1016/j.caeai.2026.100548

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics Dataset (OULAD).* Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

## Responsible-use note

A risk flag should trigger a supportive human review, not a punitive decision. Results can reflect differences in courses, data availability and student circumstances. Validate false-negative rates and subgroup performance before any operational use; the OULAD institutional context may not generalize elsewhere.

## Code organization

The pipeline is split into small modules while retaining `run_project.py` as the single entry point. Run it the same way as before: `python run_project.py`.

| File | Responsibility |
|---|---|
| `run_project.py` | Orchestrates EDA, model training, tuning, evaluation, and artifact saving |
| `project_config.py` | Shared paths, feature lists, dataset URLs, and configuration constants |
| `data_pipeline.py` | Dataset download/loading, joins, cleaning, and early-window feature engineering |
| `eda.py` | EDA summaries, plots, and the data preprocessing/EDA report |
| `modeling.py` | Model pipelines, metrics, evaluation figures, and SHAP/importance analysis |
| `reports.py` | Model evaluation report generation |
| `predict_risk.py` | Inference using the saved model artifact |

This is an organizational refactor: existing function bodies and modeling choices are preserved. It does not by itself change model quality or resolve dataset limitations.
