# Phase 5A — Training-only diagnosis and model comparison

## Data protection

- Read only `data/processed/train.csv` (80,000 rows). This script does not open `data/processed/test.csv`.
- CV used 24,000 rows sampled stratified from the training split, with 3-fold StratifiedKFold and `random_state=42`.
- The test set was not used for diagnostics, feature selection, model comparison, tuning, or refitting.
- Existing Phase 3 model artifact was not overwritten.

## Key diagnosis

- Training target counts: {'Placed': 43567, 'Not Placed': 36433}. Placed prevalence: 54.459%.
- Numeric feature with strongest absolute Pearson association: `backlogs` (r=-0.0845).
- Numeric correlations and categorical placement rates are descriptive associations, not evidence of causation.
- Most individual numeric correlations are small, indicating weak simple linear associations in this dataset.
- Categorical placement rates are also close to the overall placement prevalence; inspect `phase5a_categorical_placement_rates.csv`.
- Predicting the majority class for every row gives training F1 for Placed=0.7052; this illustrates why F1 must be read alongside recall, precision, ROC-AUC, and a baseline.

## Training-only cross-validation comparison

| Model | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.5722 | 0.5812 | 0.7674 | 0.6615 | 0.5822 |
| Linear SVM | 0.5720 | 0.5809 | 0.7682 | 0.6616 | 0.5822 |
| Logistic Regression (class_weight=balanced) | 0.5613 | 0.6040 | 0.5646 | 0.5836 | 0.5822 |
| HistGradientBoosting | 0.5641 | 0.5716 | 0.7969 | 0.6656 | 0.5680 |
| Random Forest | 0.5601 | 0.5668 | 0.8151 | 0.6687 | 0.5616 |

## Interpretation and next decision

- Highest mean ROC-AUC in this screening run: **Logistic Regression (baseline)** (ROC-AUC=0.5822, F1=0.6615).
- Baseline Logistic Regression: ROC-AUC=0.5822, F1=0.6615.
- The screening used a training-only 24,000-row subset to keep runtime manageable. Small differences should not be treated as decisive; run a more robust training-only comparison before changing the deployment candidate.
- `salary_package_lpa` is a post-outcome field and excluded. `student_id` is an identifier and excluded. `age` and `gender` remain excluded from model inputs by the current project policy, though gender was reviewed descriptively; excluding them does not prove fairness.
- Dataset provenance and label-generation process are unknown from the available project files. Weak results could reflect weak predictors, noisy/synthetic labels, or omitted variables; this analysis cannot establish which explanation is correct.
- No model was selected for deployment or refit in this phase. No test-set metrics were calculated. The previously reported Phase 4 evaluation belongs only to the earlier Logistic Regression artifact.

## Artifacts

- `reports/phase5a_training_cv_comparison.csv`
- `reports/phase5a_numeric_target_associations.csv`
- `reports/phase5a_classwise_numeric_means.csv`
- `reports/phase5a_categorical_placement_rates.csv`
- `reports/figures/phase5a_numeric_target_associations.png`
- `reports/phase5a_diagnostics.json`
