# Student Placement Prediction System

A beginner-friendly, reproducible ML project. The original CSV is included in `data/raw/`, so the project can be extracted and run without manually creating folders or writing code.

## Current verified status
- **Phase 1 — Dataset audit:** completed and script-tested.
- **Phase 2 — Train/test split and EDA:** completed and script-tested.
- **Phase 3 — Baseline model training and selection:** completed and run-tested. Four models plus a dummy reference baseline were compared with stratified 3-fold CV on a 24,000-row subset of the training split; the selected pipeline was refit on all 80,000 training rows. The test set was not used.
- **Phase 4 — Final evaluation and explainability:** pending Phase 3 approval/results.
- **Phase 5 — Streamlit app:** pending.

## Easiest setup on Windows
1. Extract the ZIP to a simple path, for example `C:\\StudentPlacementPrediction`.
2. Open the extracted `student-placement-prediction` folder.
3. Double-click `setup_windows.bat`. It creates a local Python virtual environment and installs the listed dependencies. Keep the window open until it says setup is complete.
4. Double-click `run_phase2_windows.bat` to re-run the dataset audit, stratified split, and EDA. The original CSV is read-only from the scripts' point of view and is not rewritten.
5. Open `reports/phase2_eda_findings.md` to read the EDA findings. Charts are in `reports/figures/`.
6. Double-click `run_phase3_windows.bat` to compare baseline models and save the selected model. Open `reports/phase3_model_selection.md` for the results.

If Windows blocks a `.bat` file, open Command Prompt in this folder and run:

```bat
setup_windows.bat
run_phase2_windows.bat
```

Python 3.11+ is recommended. Confirm Python is available by opening Command Prompt and entering `python --version`. If `python` is not recognized, install Python from python.org and enable **Add python.exe to PATH** during setup.

## What is included
- `data/raw/student_placement_prediction_dataset_2026.csv`: original dataset supplied for this project.
- `data/processed/train.csv` and `data/processed/test.csv`: reproducible stratified 80:20 split (`random_state=42`).
- `phase1_audit.py`: checks shape, columns, missingness, duplicates, target counts, and configuration consistency.
- `phase2_split_eda.py`: verifies Phase 1 artifacts, creates the split and training-only EDA outputs.
- `phase3_train_select.py`: compares Logistic Regression, Random Forest, Linear SVM, and KNN against a majority-class dummy baseline; cross-validation and final refit use training data only.
- `config.py`: central project paths, target mapping, and feature policy.
- `reports/`: JSON audit/summary, CSV tables, written findings, and charts. Phase 3 outputs are `phase3_cv_results.csv`, `phase3_model_selection.json`, and `phase3_model_selection.md`.
- `models/selected_placement_model.joblib`: fitted selected pipeline. It is not yet a final production model; Phase 4 must evaluate it once on the held-out test set.

## Modeling and fairness policy
- Target is `placement_status`; `Not Placed` maps to 0 and `Placed` maps to 1.
- `student_id` is excluded because it is an identifier.
- `salary_package_lpa` is excluded because it is known after placement and would leak the answer.
- `age` and `gender` are kept for descriptive/fairness review only and excluded from model inputs for this project. The future app must use the same configured feature list and must not invent default values for excluded fields.
- EDA associations are descriptive, not evidence of causation. Dataset provenance has not been independently established.
- The test set is reserved for later evaluation. The held-out test set remains reserved for Phase 4 and must not be used for model selection or tuning.

## Manual commands (optional)
Run these from this project folder after setup:

```bat
.venv\\Scripts\\python.exe phase1_audit.py
.venv\\Scripts\\python.exe phase2_split_eda.py
```

## Outputs
- `reports/phase1_audit.json`
- `reports/phase2_eda_summary.json`
- `reports/phase2_eda_findings.md`
- CSV analysis tables and PNG charts under `reports/`

## Phase 3 interpretation note

The baseline results show weak separation between placed and not-placed students (best cross-validated ROC-AUC is about 0.58). The majority-class dummy baseline has a higher F1 for the Placed class simply because it predicts every student as Placed; therefore F1 alone is misleading here. Logistic Regression was selected among the four actual model candidates by ROC-AUC, not by F1. These are training-only cross-validation estimates, not final test scores or reliable real-world placement probabilities.

## Phase 4 — Held-out test evaluation

Phase 4 has already been run once in the prepared project bundle. Review these generated outputs rather than rerunning the held-out evaluation:

- `reports/phase4_test_evaluation.md` — final test metrics, confusion matrix, coefficient interpretation, limitations, and validation checks.
- `reports/phase4_test_metrics.json` — machine-readable metrics and protocol metadata.
- `reports/phase4_test_predictions.csv` — predictions/probabilities for the reserved test records.
- `reports/phase4_confusion_matrix.csv` and `reports/figures/phase4_confusion_matrix.png`.
- `reports/phase4_logistic_coefficients.csv` and `reports/figures/phase4_logistic_coefficients.png`.
- `phase4_evaluate.py` — source code for the one-time evaluation, with safeguards against overwriting an existing result.

The selected Phase 3 pipeline was loaded as-is. Phase 4 does not fit or tune a model. The test file SHA-256 was identical before and after evaluation. The observed test ROC-AUC was approximately 0.585, indicating weak discrimination; do not present this baseline as a reliable real-world placement predictor.


## Phase 5A — Training-only diagnosis (run this, not Phase 4 again)

Phase 5A investigates weak performance and screens a few alternatives using only the training split. It does **not** read `data/processed/test.csv`, does **not** overwrite the Phase 3 model, and does **not** calculate new test metrics.

After `setup_windows.bat` has completed, double-click `run_phase5a_windows.bat`. Open `reports/phase5a_diagnostics.md` when it finishes.

Phase 5A outputs:
- `reports/phase5a_diagnostics.md` — findings and model comparison.
- `reports/phase5a_diagnostics.json` — machine-readable diagnostics and safeguards.
- `reports/phase5a_training_cv_comparison.csv` — training-only cross-validation scores.
- `reports/phase5a_numeric_target_associations.csv` — descriptive numeric correlations with target.
- `reports/phase5a_classwise_numeric_means.csv` — class-wise feature means.
- `reports/phase5a_categorical_placement_rates.csv` — placement rates by category.
- `reports/figures/phase5a_numeric_target_associations.png` — correlation chart.

The comparison is a practical screening run on a deterministic stratified 24,000-row subset of the training split with 3-fold stratified CV. It is not an exhaustive hyperparameter search. The test set had already been evaluated once in Phase 4 for the earlier Logistic Regression artifact; it must not be reused to compare or tune newly screened models. If a different candidate is chosen later, its generalization performance cannot be claimed from the Phase 4 metrics.

- `reports/phase5a_validation.json` — automated checks for output ranges, script safeguards, and test-file integrity.


## Phase 5B — Training-only feature engineering and provenance review

Run `run_phase5b_windows.bat` to compare transparent engineered features using only the training split. This does not read `test.csv`, does not tune against test data, and does not overwrite the saved model. Review `reports/phase5b_feature_engineering.md`. Dataset provenance and label-generation logic remain unverified because the supplied overview does not identify a trusted source or label creation rule.

## Run the Streamlit demo

1. Run `setup_windows.bat` if the project environment is not already installed. It installs the dependencies listed in `requirements.txt`, including Streamlit.
2. Double-click `run_app_windows.bat`.
3. A browser tab should open with the Student Placement Prediction System. If it does not, copy the local URL printed in the terminal into your browser.
4. Complete all fields; the app intentionally does not prefill a student profile.

**Limitations:** the current Logistic Regression model has weak test discrimination (ROC-AUC about 0.585), and dataset provenance is unverified. This app is an educational prototype only and must not be used to rank, exclude, or make high-stakes decisions about students. The score is not a placement guarantee.

## Final local validation

After `setup_windows.bat` completes, double-click `run_validation_windows.bat`. This runs syntax, required-file, split-schema, target-label, feature-exclusion, and one-row saved-model inference checks. It does not fit or tune a model and does not calculate new metrics on the held-out test set. The result is written to `reports/final_project_validation.json`.

The project bundle's Python syntax and model inference checks pass in the preparation environment. The Streamlit UI itself could not be launched in that environment because Streamlit is not installed there and package downloads were unavailable; the included setup script installs Streamlit on the user's computer. Run `run_app_windows.bat` locally to verify the browser UI.
