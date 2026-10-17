# Phase 4 — Held-out Test Evaluation

## Evaluation protocol

- Evaluated the already-selected `Logistic Regression` pipeline once on `data/processed/test.csv` (20,000 rows).
- Loaded the saved Phase 3 artifact; no fitting, model selection, feature selection, or tuning occurred in this phase.
- Test set was not used in Phase 3 model selection. The test CSV SHA-256 matched before and after evaluation.
- Positive class: `Placed`; classification threshold: 0.5.
- Excluded model inputs: `student_id`, `salary_package_lpa`, `age`, `gender`.

## Final test metrics

| Metric | Test result |
|---|---:|
| Accuracy | 0.5706 (57.06%) |
| Precision (Placed) | 0.5793 |
| Recall (Placed) | 0.7733 |
| F1 (Placed) | 0.6624 |
| ROC-AUC | 0.5850 |
| Balanced accuracy | 0.5508 |
| Matthews correlation coefficient | 0.1136 |

## Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual \ Predicted | Not Placed | Placed |
|---|---:|---:|
| Not Placed | TN = 2990 | FP = 6118 |
| Placed | FN = 2469 | TP = 8423 |

- False positives: 6118 students predicted as placed who were not placed in the dataset.
- False negatives: 2469 students predicted as not placed who were placed in the dataset.

## Coefficient-based feature interpretation

The coefficient file ranks transformed features by absolute logistic-regression coefficient. Positive coefficients are associated with higher model log-odds of `Placed`; negative coefficients with lower log-odds, conditional on the model's other inputs. Numeric features were standardized and categorical features one-hot encoded.
These are model associations, not causal effects. Coefficient magnitude depends on encoding and scaling; correlated predictors can share or distort apparent importance. Do not treat this ranking as proof that changing a feature will change placement outcomes.

### Top 10 by absolute coefficient

| Transformed feature | Coefficient |
|---|---:|
| `numeric__backlogs` | -0.17269 |
| `numeric__internships_count` | 0.12833 |
| `numeric__projects_count` | 0.12631 |
| `numeric__coding_skill_score` | 0.10633 |
| `numeric__mock_interview_score` | 0.09423 |
| `numeric__logical_reasoning_score` | 0.07387 |
| `numeric__aptitude_score` | 0.06647 |
| `numeric__communication_skill_score` | 0.06209 |
| `categorical__volunteer_experience_Yes` | 0.04722 |
| `numeric__leadership_score` | 0.04660 |

## Limitations and responsible use

- This is a single evaluation of one previously selected model on one held-out split; it is not a guarantee of future performance.
- The dataset's real-world provenance and representativeness have not been independently established.
- Excluding age and gender does not guarantee fairness because other predictors may be proxies.
- The target is historical placement status; model associations do not establish causation.
- Predicted probabilities may not be calibrated and must not be presented as guaranteed individual outcomes.
- This is a baseline classifier, not a hiring or admissions decision system. Avoid using it to make consequential decisions about individual students.
- Before real deployment, validate provenance, leakage, representativeness, calibration, subgroup performance, and prospective performance on independently collected data.

## Artifacts

- `reports/phase4_test_metrics.json` — metrics, target counts, hashes, protocol metadata.
- `reports/phase4_test_predictions.csv` — held-out predictions and predicted probabilities.
- `reports/phase4_confusion_matrix.csv` and `reports/figures/phase4_confusion_matrix.png`.
- `reports/phase4_logistic_coefficients.csv` and `reports/figures/phase4_logistic_coefficients.png`.

## Automated validation

- Test rows: 20,000; prediction count: 20,000.
- Probability array shape: [20000, 2]; probability values finite: True.
- Confusion-matrix total equals test rows: True.
- Metrics finite: True.
- Test file unchanged: True.
- Model fit/tuning in Phase 4: no.
