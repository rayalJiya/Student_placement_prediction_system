# Phase 5B — Training-only feature engineering and provenance review

## Data protection

- Read only `data/processed/train.csv` (80,000 rows). `test.csv` was not opened.
- Comparison used 24,000 stratified training rows and 3-fold StratifiedKFold (random_state=42).
- All feature engineering is inside each cross-validation pipeline, so summaries are recomputed from each fold's input rows.
- No model was refit for deployment, no saved model was overwritten, and no test metrics were calculated.

## Provenance and label-generation review

- The project overview describes `placement_status` as the target and `salary_package_lpa` as a post-placement outcome.
- The provided project files do not identify a trusted publisher, collection period, data dictionary beyond column descriptions, or the rule used to create placement labels.
- Therefore dataset provenance and label-generation process remain **unverified**. Do not describe the dataset as real-world or synthetic without confirmation from its provider.
- `salary_package_lpa` is excluded because it is known after the placement outcome and would leak outcome information.

## Training-only comparison

| Candidate | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + engineered summaries | 0.5712 | 0.5804 | 0.7669 | 0.6608 | 0.5822 |
| Logistic Regression baseline | 0.5722 | 0.5812 | 0.7674 | 0.6615 | 0.5822 |
| HistGradientBoosting + engineered summaries | 0.5642 | 0.5717 | 0.7969 | 0.6657 | 0.5707 |

## Interpretation

- The two Logistic Regression candidates are tied at ROC-AUC=0.5822 when rounded to four decimals; the baseline has slightly higher F1. This is not a meaningful improvement.
- Engineered Logistic Regression ROC-AUC change vs baseline: +0.0000; F1 change: -0.0007.
- Engineered summaries are transparent combinations of existing pre-outcome fields; they do not add new information and may be redundant for linear models.
- Small cross-validation differences on a 24,000-row subset are screening evidence only, not proof of a robust improvement.
- No model change is recommended from this phase alone. Preserve the existing model and use the current results as research diagnostics.

## Proposed engineered features

- `technical_academic_mean`: mean of CGPA, coding, aptitude, and logical reasoning scores.
- `experience_activity_count`: internships, projects, certifications, and hackathons summed.
- `career_presence_mean_log1p`: mean log1p of GitHub repositories and LinkedIn connections.
- `interview_readiness_mean`: mean of coding, aptitude, communication, and mock-interview scores.
- `leadership_engagement_mean`: mean of extracurricular score, leadership score, and binary volunteer indicator.

## Artifacts

- `reports/phase5b_feature_engineering_cv.csv`
- `reports/phase5b_provenance_review.json`
