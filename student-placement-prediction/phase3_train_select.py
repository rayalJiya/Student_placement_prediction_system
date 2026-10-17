"""Phase 3: baseline model comparison using training data only.

This script does NOT score or inspect the held-out test set. It compares four
pipelines using stratified 3-fold cross-validation on data/processed/train.csv,
then refits the selected pipeline on the full training split and saves it.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC

import config

POSITIVE_LABEL = "Placed"
NEGATIVE_LABEL = "Not Placed"
N_SPLITS = 3
CV_SAMPLE_SIZE = 24000  # stratified training-only subset to keep baseline comparison feasible on one CPU core

MODEL_DIR = config.PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CV_RESULTS_PATH = config.REPORTS_DIR / "phase3_cv_results.csv"
SUMMARY_PATH = config.REPORTS_DIR / "phase3_model_selection.json"
FINDINGS_PATH = config.REPORTS_DIR / "phase3_model_selection.md"
MODEL_PATH = MODEL_DIR / "selected_placement_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, config.NUMERICAL_FEATURES),
        ("categorical", categorical_pipe, config.CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    # All preprocessing is inside each pipeline to prevent fold-level leakage.
    return {
        "Dummy Baseline (majority class)": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=40, max_depth=12, min_samples_leaf=5, max_samples=0.70,
                random_state=config.RANDOM_STATE, n_jobs=1,
                class_weight=None,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LinearSVC(
                C=1.0, dual="auto", random_state=config.RANDOM_STATE,
                max_iter=5000,
            )),
        ]),
        "KNN": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=1)),
        ]),
    }


def main() -> None:
    started = time.time()
    print("Phase 3: checking input files and feature policy...", flush=True)
    if not config.TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training split missing: {config.TRAIN_DATA_PATH}. Run Phase 2 first.")
    if not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test split missing: {config.TEST_DATA_PATH}. Run Phase 2 first.")

    train = pd.read_csv(config.TRAIN_DATA_PATH)
    # Deliberately do not read test.csv: Phase 3 is not permitted to use it.
    expected = set(config.MODEL_FEATURES + [config.TARGET_COLUMN])
    missing = sorted(expected - set(train.columns))
    if missing:
        raise ValueError(f"Training split is missing required columns: {missing}")
    forbidden = set(config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS + config.EXCLUDED_FAIRNESS_COLUMNS)
    accidental = forbidden.intersection(config.MODEL_FEATURES)
    if accidental:
        raise ValueError(f"Forbidden model features detected: {sorted(accidental)}")
    labels = set(train[config.TARGET_COLUMN].dropna().unique())
    if labels != {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise ValueError(f"Unexpected target labels: {labels}")
    if train[config.MODEL_FEATURES + [config.TARGET_COLUMN]].isna().all(axis=0).any():
        raise ValueError("At least one required training column contains only missing values.")

    X = train[config.MODEL_FEATURES].copy()
    y = train[config.TARGET_COLUMN].copy()
    # Stratified subset drawn only from training split. This makes four-model CV practical
    # on a single CPU while the selected pipeline is still refit on all training rows.
    if len(train) > CV_SAMPLE_SIZE:
        X_cv, _, y_cv, _ = train_test_split(
            X, y, train_size=CV_SAMPLE_SIZE, stratify=y,
            random_state=config.RANDOM_STATE,
        )
    else:
        X_cv, y_cv = X, y
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    scorers = {
        "accuracy": "accuracy",
        "precision_placed": make_scorer(precision_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_placed": make_scorer(recall_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_placed": make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "roc_auc": "roc_auc",
    }

    all_results = []
    model_objects = build_models()
    for name, pipeline in model_objects.items():
        print(f"\nCross-validating {name} ({N_SPLITS} folds)...", flush=True)
        model_started = time.time()
        cv_result = cross_validate(
            pipeline, X_cv, y_cv, cv=cv, scoring=scorers,
            n_jobs=1, return_train_score=False, error_score="raise",
        )
        row = {
            "model": name,
            "cv_folds": N_SPLITS,
            "fit_seconds_total": round(float(np.sum(cv_result["fit_time"])), 3),
            "score_seconds_total": round(float(np.sum(cv_result["score_time"])), 3),
        }
        for metric in scorers:
            values = cv_result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        all_results.append(row)
        print(
            f"Finished {name}: F1={row['f1_placed_mean']:.4f} ± {row['f1_placed_std']:.4f}; "
            f"ROC-AUC={row['roc_auc_mean']:.4f}; elapsed={time.time()-model_started:.1f}s",
            flush=True,
        )

    results = pd.DataFrame(all_results)
    results.to_csv(CV_RESULTS_PATH, index=False)
    # Dummy is a reference baseline only, never a candidate for the selected deployable model.
    candidate_results = results[results["model"] != "Dummy Baseline (majority class)"].sort_values(
        ["roc_auc_mean", "f1_placed_mean", "accuracy_mean"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(candidate_results.loc[0, "model"])
    selected_pipeline = model_objects[selected_name]

    print(f"\nSelected by highest mean cross-validated ROC-AUC: {selected_name}", flush=True)
    print("Refitting selected pipeline on all 80,000 training rows (test set remains untouched)...", flush=True)
    selected_pipeline.fit(X, y)
    joblib.dump(selected_pipeline, MODEL_PATH, compress=3)

    selected_row = results.loc[results["model"] == selected_name].iloc[0].to_dict()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 3 - baseline model training and selection",
        "sklearn_version": sklearn.__version__,
        "training_rows": int(len(train)),
        "cv_rows": int(len(X_cv)),
        "cv_sampling": "stratified random subset from training split only" if len(X_cv) < len(train) else "full training split",
        "training_target_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "feature_count": len(config.MODEL_FEATURES),
        "features": config.MODEL_FEATURES,
        "excluded_from_model_features": config.MODEL_EXCLUDED_COLUMNS + [config.TARGET_COLUMN],
        "positive_label": POSITIVE_LABEL,
        "cv": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "cv_rows": int(len(X_cv)),
            "cv_sampling": "stratified random subset of training split only; final selected model is refit on all training rows",
            "shuffle": True,
            "random_state": config.RANDOM_STATE,
            "selection_metric": "mean cross-validated ROC-AUC among non-dummy models; ties broken by F1 for Placed then accuracy",
            "test_set_accessed": False,
        },
        "models_compared": list(model_objects.keys()),
        "selected_model": selected_name,
        "selected_model_cv_metrics": selected_row,
        "model_artifact": str(MODEL_PATH.relative_to(config.PROJECT_ROOT)),
        "cv_results": str(CV_RESULTS_PATH.relative_to(config.PROJECT_ROOT)),
        "elapsed_seconds_including_final_fit": round(time.time() - started, 2),
        "limitations": [
            "These are cross-validation estimates on a stratified subset of the training split, not final test-set metrics. A dummy majority-class model is included to show how much performance can be achieved without learning feature patterns.",
            "The dataset's real-world provenance has not been established; results should not be interpreted as validated real-world placement probabilities.",
            "The baseline settings are not an exhaustive hyperparameter search.",
            "A dummy majority-class baseline is included for context; F1 alone can reward predicting the majority class, so ROC-AUC is the primary selection metric.",
            "Excluding age and gender does not by itself guarantee fairness because other features may act as proxies.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps({
        "selected_model": selected_name,
        "target_column": config.TARGET_COLUMN,
        "target_mapping": config.TARGET_MAPPING,
        "model_features": config.MODEL_FEATURES,
        "excluded_columns": config.MODEL_EXCLUDED_COLUMNS,
        "training_rows": int(len(train)),
        "selection_metric": "mean 3-fold cross-validated ROC-AUC among non-dummy models; ties broken by F1 for Placed then accuracy",
        "test_set_used_during_selection": False,
    }, indent=2), encoding="utf-8")

    lines = [
        "# Phase 3 — Baseline Model Training and Selection", "",
        "## Data safeguards", "",
        f"- Used only `data/processed/train.csv` ({len(train):,} rows). The script deliberately does not read `test.csv`.",
        f"- Stratified {N_SPLITS}-fold cross-validation on {len(X_cv):,} training-only rows; shuffle enabled; `random_state={config.RANDOM_STATE}`.",
        "- For runtime feasibility on a single CPU, model screening uses a stratified 24,000-row subset of the training split; the selected pipeline is refit on all 80,000 training rows.",
        "- A majority-class dummy model is included as a reference baseline; it is not eligible for selection.",
        f"- Target positive class: `{POSITIVE_LABEL}`.",
        f"- Candidate model features ({len(config.MODEL_FEATURES)}): " + ", ".join(f"`{c}`" for c in config.MODEL_FEATURES) + ".",
        "- Excluded `student_id`, `salary_package_lpa`, `age`, and `gender` from model inputs.",
        "- Imputation, scaling, and one-hot encoding are inside each pipeline, so they are fitted independently within each CV fold.",
        "", "## Cross-validation results", "",
        "Values below are means across folds; standard deviations are in parentheses. They are not test-set results. The dummy model predicts the training majority class and is a no-learning reference.", "",
        "| Model | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) | ROC-AUC |", "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['model']} | {r['accuracy_mean']:.4f} ({r['accuracy_std']:.4f}) | "
            f"{r['precision_placed_mean']:.4f} ({r['precision_placed_std']:.4f}) | "
            f"{r['recall_placed_mean']:.4f} ({r['recall_placed_std']:.4f}) | "
            f"{r['f1_placed_mean']:.4f} ({r['f1_placed_std']:.4f}) | "
            f"{r['roc_auc_mean']:.4f} ({r['roc_auc_std']:.4f}) |"
        )
    lines += [
        "", "## Selection decision", "",
        f"Selected model: **{selected_name}**, based on highest mean cross-validated ROC-AUC among the four real model candidates; ties are broken by F1 for the `Placed` class and then accuracy. The dummy majority-class baseline is included only as a reference.",
        "The selected pipeline has been refitted on the complete training split and saved as `models/selected_placement_model.joblib`.",
        "", "## What this does not claim", "",
        "- The held-out test set has not been used or scored in Phase 3.",
        "- CV scores are not final test performance and do not establish causal relationships.",
        "- The dummy majority-class baseline is for reference only; model selection uses ROC-AUC as the primary metric because F1 alone can favor predicting the majority class.",
        "- Baseline hyperparameters have not been exhaustively tuned.",
        "- Dataset provenance is not established; don't present these numbers as guaranteed real-world placement prediction accuracy.",
        "", "## Artifacts", "",
        "- `reports/phase3_cv_results.csv` — sortable model comparison.",
        "- `reports/phase3_model_selection.json` — run metadata and selected metrics.",
        "- `models/selected_placement_model.joblib` — fitted pipeline.",
        "- `models/model_metadata.json` — features and target details for downstream consistency checks.",
        "", f"Total elapsed time including final refit: {time.time()-started:.1f} seconds.", "",
    ]
    FINDINGS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\nPhase 3 complete. Test set was not used.", flush=True)
    print(f"Results: {CV_RESULTS_PATH}", flush=True)
    print(f"Selected model: {selected_name}", flush=True)
    print(f"Elapsed: {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
