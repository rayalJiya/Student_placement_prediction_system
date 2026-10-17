"""Phase 5A: training-only diagnostics and candidate model comparison.

IMPORTANT:
- Reads data/processed/train.csv only.
- Does not read test.csv.
- Does not overwrite the Phase 3 selected model.
- Compares candidates using stratified CV on a deterministic training-only subset.
"""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
import config

TARGET = config.TARGET_COLUMN
POSITIVE = "Placed"
NEGATIVE = "Not Placed"
CV_SAMPLE_SIZE = 24000
N_SPLITS = 3
RANDOM_STATE = config.RANDOM_STATE
OUT = config.REPORTS_DIR
FIG = config.FIGURES_DIR
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def make_preprocessor(dense=False, scale_numeric=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, config.NUMERICAL_FEATURES),
        ("categorical", cat_pipe, config.CATEGORICAL_FEATURES),
    ])

def main():
    start = time.time()
    train_path = config.TRAIN_DATA_PATH
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}. Run Phase 2 first.")
    # Deliberately never open TEST_DATA_PATH in this script.
    df = pd.read_csv(train_path)
    expected = set(config.MODEL_FEATURES + [TARGET])
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")
    if set(df[TARGET].dropna().unique()) != {NEGATIVE, POSITIVE}:
        raise ValueError(f"Unexpected target values: {sorted(df[TARGET].dropna().unique())}")
    forbidden = set(config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS)
    if forbidden.intersection(config.MODEL_FEATURES):
        raise ValueError("ID/post-outcome columns incorrectly included in model features")

    X = df[config.MODEL_FEATURES].copy()
    y = df[TARGET].copy()
    # Numeric target association (descriptive only; not causal).
    y01 = (y == POSITIVE).astype(int)
    num_cols = [c for c in config.EDA_NUMERICAL_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    target_corr = df[num_cols].corrwith(y01).sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr.rename("pearson_correlation_with_placed").to_csv(OUT / "phase5a_numeric_target_associations.csv")
    class_means = df.groupby(TARGET)[num_cols].mean().T
    class_means["difference_placed_minus_not_placed"] = class_means[POSITIVE] - class_means[NEGATIVE]
    class_means.to_csv(OUT / "phase5a_classwise_numeric_means.csv")

    categorical_cols = [c for c in config.EDA_CATEGORICAL_COLUMNS if c in df.columns]
    categorical_rows = []
    for col in categorical_cols:
        rates = df.groupby(col, dropna=False)[TARGET].apply(lambda s: (s == POSITIVE).mean())
        counts = df.groupby(col, dropna=False)[TARGET].size()
        for level in rates.index:
            categorical_rows.append({"feature": col, "level": str(level), "rows": int(counts.loc[level]),
                                     "placed_rate": float(rates.loc[level])})
    cat_rates = pd.DataFrame(categorical_rows)
    cat_rates.to_csv(OUT / "phase5a_categorical_placement_rates.csv", index=False)

    # Check whether a trivial majority prediction can appear competitive on F1.
    majority_label = y.value_counts().idxmax()
    majority_f1 = float(f1_score(y, np.repeat(majority_label, len(y)), pos_label=POSITIVE, zero_division=0))
    target_counts = {str(k): int(v) for k, v in y.value_counts().items()}

    # Deterministic stratified subset of training rows for practical one-core screening.
    if len(df) > CV_SAMPLE_SIZE:
        X_cv, _, y_cv, _ = train_test_split(X, y, train_size=CV_SAMPLE_SIZE, stratify=y, random_state=RANDOM_STATE)
    else:
        X_cv, y_cv = X, y
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "accuracy": "accuracy",
        "precision_placed": make_scorer(precision_score, pos_label=POSITIVE, zero_division=0),
        "recall_placed": make_scorer(recall_score, pos_label=POSITIVE, zero_division=0),
        "f1_placed": make_scorer(f1_score, pos_label=POSITIVE, zero_division=0),
        "roc_auc": "roc_auc",
    }
    models = {
        "Logistic Regression (baseline)": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression (class_weight=balanced)": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("model", LinearSVC(C=1.0, dual="auto", random_state=RANDOM_STATE, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=5,
                max_samples=0.7, random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocess", make_preprocessor(dense=True, scale_numeric=False)),
            ("model", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08,
                max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE)),
        ]),
    }
    result_rows = []
    for name, pipe in models.items():
        print(f"CV: {name}", flush=True)
        t0 = time.time()
        scores = cross_validate(pipe, X_cv, y_cv, cv=cv, scoring=scorers,
                                n_jobs=1, return_train_score=False, error_score="raise")
        row = {"model": name, "cv_folds": N_SPLITS, "cv_rows": int(len(X_cv)),
               "fit_seconds_total": round(float(scores["fit_time"].sum()), 3),
               "score_seconds_total": round(float(scores["score_time"].sum()), 3)}
        for metric in scorers:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        result_rows.append(row)
        print(f"  ROC-AUC={row['roc_auc_mean']:.4f}; F1={row['f1_placed_mean']:.4f}; "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
    results = pd.DataFrame(result_rows).sort_values(["roc_auc_mean", "f1_placed_mean"], ascending=False)
    results.to_csv(OUT / "phase5a_training_cv_comparison.csv", index=False)

    # Plot strongest training-only feature correlations.
    top = target_corr.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax)
    ax.axvline(0, linewidth=0.8)
    ax.set_title("Training-only numeric feature associations with placement")
    ax.set_xlabel("Pearson correlation with Placed=1")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG / "phase5a_numeric_target_associations.png", dpi=150)
    plt.close(fig)

    best = results.iloc[0].to_dict()
    baseline_row = results[results["model"] == "Logistic Regression (baseline)"].iloc[0].to_dict()
    findings = [
        "# Phase 5A — Training-only diagnosis and model comparison", "",
        "## Data protection", "",
        f"- Read only `data/processed/train.csv` ({len(df):,} rows). This script does not open `data/processed/test.csv`.",
        f"- CV used {len(X_cv):,} rows sampled stratified from the training split, with {N_SPLITS}-fold StratifiedKFold and `random_state={RANDOM_STATE}`.",
        "- The test set was not used for diagnostics, feature selection, model comparison, tuning, or refitting.",
        "- Existing Phase 3 model artifact was not overwritten.",
        "", "## Key diagnosis", "",
        f"- Training target counts: {target_counts}. Placed prevalence: {(y == POSITIVE).mean():.3%}.",
        f"- Numeric feature with strongest absolute Pearson association: `{target_corr.index[0]}` (r={target_corr.iloc[0]:+.4f}).",
        "- Numeric correlations and categorical placement rates are descriptive associations, not evidence of causation.",
        "- Most individual numeric correlations are small, indicating weak simple linear associations in this dataset.",
        "- Categorical placement rates are also close to the overall placement prevalence; inspect `phase5a_categorical_placement_rates.csv`.",
        f"- Predicting the majority class for every row gives training F1 for Placed={majority_f1:.4f}; this illustrates why F1 must be read alongside recall, precision, ROC-AUC, and a baseline.",
        "", "## Training-only cross-validation comparison", "",
        "| Model | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in results.iterrows():
        findings.append(f"| {r['model']} | {r['accuracy_mean']:.4f} | {r['precision_placed_mean']:.4f} | {r['recall_placed_mean']:.4f} | {r['f1_placed_mean']:.4f} | {r['roc_auc_mean']:.4f} |")
    findings += [
        "", "## Interpretation and next decision", "",
        f"- Highest mean ROC-AUC in this screening run: **{best['model']}** (ROC-AUC={best['roc_auc_mean']:.4f}, F1={best['f1_placed_mean']:.4f}).",
        f"- Baseline Logistic Regression: ROC-AUC={baseline_row['roc_auc_mean']:.4f}, F1={baseline_row['f1_placed_mean']:.4f}.",
        "- The screening used a training-only 24,000-row subset to keep runtime manageable. Small differences should not be treated as decisive; run a more robust training-only comparison before changing the deployment candidate.",
        "- `salary_package_lpa` is a post-outcome field and excluded. `student_id` is an identifier and excluded. `age` and `gender` remain excluded from model inputs by the current project policy, though gender was reviewed descriptively; excluding them does not prove fairness.",
        "- Dataset provenance and label-generation process are unknown from the available project files. Weak results could reflect weak predictors, noisy/synthetic labels, or omitted variables; this analysis cannot establish which explanation is correct.",
        "- No model was selected for deployment or refit in this phase. No test-set metrics were calculated. The previously reported Phase 4 evaluation belongs only to the earlier Logistic Regression artifact.",
        "", "## Artifacts", "",
        "- `reports/phase5a_training_cv_comparison.csv`",
        "- `reports/phase5a_numeric_target_associations.csv`",
        "- `reports/phase5a_classwise_numeric_means.csv`",
        "- `reports/phase5a_categorical_placement_rates.csv`",
        "- `reports/figures/phase5a_numeric_target_associations.png`",
        "- `reports/phase5a_diagnostics.json`",
    ]
    (OUT / "phase5a_diagnostics.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    payload = {
        "phase": "Phase 5A training-only diagnosis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "training_rows": int(len(df)),
        "cv_rows": int(len(X_cv)),
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "test_set_read": False,
        "test_set_used_for_tuning": False,
        "existing_phase3_model_overwritten": False,
        "target_counts": target_counts,
        "placed_prevalence": float((y == POSITIVE).mean()),
        "majority_class_training_f1_for_placed": majority_f1,
        "numeric_target_associations": {str(k): float(v) for k, v in target_corr.items()},
        "best_training_cv_candidate": str(best["model"]),
        "best_training_cv_roc_auc": float(best["roc_auc_mean"]),
        "baseline_logistic_cv_roc_auc": float(baseline_row["roc_auc_mean"]),
        "limitations": [
            "Cross-validation uses a stratified 24,000-row subset of training split for runtime feasibility.",
            "Candidate comparison is screening, not exhaustive hyperparameter optimization.",
            "The test set is intentionally not opened.",
            "Dataset provenance and label-generation process are unknown.",
            "Association does not imply causation; excluding protected/sensitive attributes does not guarantee fairness."
        ],
        "elapsed_seconds": round(time.time() - start, 2)
    }
    (OUT / "phase5a_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nPhase 5A complete. Best training CV candidate: {best['model']} ROC-AUC={best['roc_auc_mean']:.4f}", flush=True)
    print(f"Outputs: {OUT / 'phase5a_diagnostics.md'}", flush=True)

if __name__ == "__main__":
    main()
