# Phase 3 — Baseline Model Training and Selection

## Data safeguards

- Used only `data/processed/train.csv` (80,000 rows). The script deliberately does not read `test.csv`.
- Stratified 3-fold cross-validation on 24,000 training-only rows; shuffle enabled; `random_state=42`.
- For runtime feasibility on a single CPU, model screening uses a stratified 24,000-row subset of the training split; the selected pipeline is refit on all 80,000 training rows.
- A majority-class dummy model is included as a reference baseline; it is not eligible for selection.
- Target positive class: `Placed`.
- Candidate model features (21): `cgpa`, `branch`, `college_tier`, `internships_count`, `projects_count`, `certifications_count`, `coding_skill_score`, `aptitude_score`, `communication_skill_score`, `logical_reasoning_score`, `hackathons_participated`, `github_repos`, `linkedin_connections`, `mock_interview_score`, `attendance_percentage`, `backlogs`, `extracurricular_score`, `leadership_score`, `volunteer_experience`, `sleep_hours`, `study_hours_per_day`.
- Excluded `student_id`, `salary_package_lpa`, `age`, and `gender` from model inputs.
- Imputation, scaling, and one-hot encoding are inside each pipeline, so they are fitted independently within each CV fold.

## Cross-validation results

Values below are means across folds; standard deviations are in parentheses. They are not test-set results. The dummy model predicts the training majority class and is a no-learning reference.

| Model | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Baseline (majority class) | 0.5446 (0.0001) | 0.5446 (0.0001) | 1.0000 (0.0000) | 0.7052 (0.0001) | 0.5000 (0.0000) |
| Logistic Regression | 0.5722 (0.0063) | 0.5812 (0.0043) | 0.7674 (0.0063) | 0.6615 (0.0051) | 0.5822 (0.0012) |
| Random Forest | 0.5583 (0.0097) | 0.5668 (0.0064) | 0.8014 (0.0082) | 0.6640 (0.0071) | 0.5589 (0.0069) |
| Linear SVM | 0.5720 (0.0067) | 0.5809 (0.0045) | 0.7682 (0.0072) | 0.6616 (0.0056) | 0.5822 (0.0012) |
| KNN | 0.5343 (0.0013) | 0.5627 (0.0011) | 0.6494 (0.0014) | 0.6030 (0.0012) | 0.5306 (0.0077) |

## Selection decision

Selected model: **Logistic Regression**, based on highest mean cross-validated ROC-AUC among the four real model candidates; ties are broken by F1 for the `Placed` class and then accuracy. The dummy majority-class baseline is included only as a reference.
The selected pipeline has been refitted on the complete training split and saved as `models/selected_placement_model.joblib`.

## What this does not claim

- The held-out test set has not been used or scored in Phase 3.
- CV scores are not final test performance and do not establish causal relationships.
- The dummy majority-class baseline is for reference only; model selection uses ROC-AUC as the primary metric because F1 alone can favor predicting the majority class.
- Baseline hyperparameters have not been exhaustively tuned.
- Dataset provenance is not established; don't present these numbers as guaranteed real-world placement prediction accuracy.

## Artifacts

- `reports/phase3_cv_results.csv` — sortable model comparison.
- `reports/phase3_model_selection.json` — run metadata and selected metrics.
- `models/selected_placement_model.joblib` — fitted pipeline.
- `models/model_metadata.json` — features and target details for downstream consistency checks.

Total elapsed time including final refit: 8.2 seconds.
