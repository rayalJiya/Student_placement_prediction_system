# Phase 2 — Train-Test Split and Exploratory Data Analysis

## Scope and reproducibility
- Source CSV: `/mnt/data/student_placement_prediction_dataset_2026.csv` (left unchanged).
- Split: stratified 80:20, `random_state=42`.
- Training rows: 80,000; test rows: 20,000.
- EDA charts and association summaries were computed from the training split only. The test split was not used to select features or inspect feature relationships.
- The saved split files omit `student_id` and `salary_package_lpa`; age and gender remain in the split files for descriptive auditing, but are excluded from model features.

## Target distribution in training data
- Not Placed: 36,433 (45.54%).
- Placed: 43,567 (54.46%).

## Numeric features with strongest absolute Pearson association with target

These are univariate correlations with `Placed=1` and `Not Placed=0`; they are not causal effects and do not capture nonlinear relationships or interactions.

- `backlogs`: r = -0.0845.
- `internships_count`: r = 0.0618.
- `projects_count`: r = 0.0605.
- `coding_skill_score`: r = 0.0514.
- `mock_interview_score`: r = 0.0460.
- `logical_reasoning_score`: r = 0.0357.
- `aptitude_score`: r = 0.0321.
- `communication_skill_score`: r = 0.0303.

## Categorical placement-rate patterns

Observed rates are descriptive and can reflect confounding or different group sizes.

- `gender`: highest observed rate `Male` (54.51%); lowest `Female` (54.38%).
- `branch`: highest observed rate `Civil` (55.01%); lowest `ECE` (53.72%).
- `college_tier`: highest observed rate `Tier 3` (54.54%); lowest `Tier 2` (54.41%).
- `volunteer_experience`: highest observed rate `Yes` (54.58%); lowest `No` (54.38%).

## Age and gender policy

- **Age:** excluded from model features. Training-set mean age by target: {'Not Placed': 21.00120769631927, 'Placed': 21.011430670002525}. It has a narrow range and little apparent univariate separation; retain only for descriptive checks unless a defensible use case is established.
- **Gender:** excluded from model features as a conservative fairness choice. Observed training placement rates by gender: {'Female': 54.38, 'Male': 54.51}. These rates should be reviewed for fairness, not treated as causal differences.
- The future application must request exactly the configured model features. It must not silently fill age/gender with arbitrary defaults.

## Leakage and quality safeguards

- `student_id` is omitted from split files because it is an identifier.
- `salary_package_lpa` is omitted because it is known after placement and directly reveals the target in this dataset.
- Split is stratified; train/test class counts are recorded in the JSON summary for verification.
- No model has been trained in this phase.

## Generated artifacts

- `data/processed/train.csv`, `data/processed/test.csv`
- `reports/phase1_audit.json`, `reports/phase2_eda_summary.json`
- `reports/numeric_classwise_summary.csv`, `reports/numeric_correlation_matrix_train.csv`, `reports/numeric_spearman_correlation_matrix_train.csv`, `reports/numeric_target_correlations_train.csv`, `reports/categorical_placement_rates_train.csv`
- Charts under `reports/figures/`.

## Limitations

The dataset's provenance is not established by this analysis. Its size alone does not prove it represents real placement outcomes. Correlation and category-level rates do not establish causation, and a random split cannot guarantee generalization to future student cohorts.
