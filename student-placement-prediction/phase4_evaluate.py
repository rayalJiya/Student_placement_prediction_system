"""Phase 4: one-time evaluation of the already-selected model on the held-out test set.

This script NEVER fits or tunes a model. It loads the Phase 3 artifact, reads the
reserved test CSV, performs one prediction/probability evaluation pass, and writes
metrics, confusion matrix, and coefficient-based diagnostics.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    balanced_accuracy_score, matthews_corrcoef,
)

import config

MODEL_PATH = config.PROJECT_ROOT / "models" / "selected_placement_model.joblib"
META_PATH = config.PROJECT_ROOT / "models" / "model_metadata.json"
REPORT_JSON = config.REPORTS_DIR / "phase4_test_metrics.json"
REPORT_MD = config.REPORTS_DIR / "phase4_test_evaluation.md"
PREDICTIONS_PATH = config.REPORTS_DIR / "phase4_test_predictions.csv"
CONFUSION_CSV = config.REPORTS_DIR / "phase4_confusion_matrix.csv"
COEFFICIENTS_CSV = config.REPORTS_DIR / "phase4_logistic_coefficients.csv"
CONFUSION_PNG = config.FIGURES_DIR / "phase4_confusion_matrix.png"
COEFFICIENTS_PNG = config.FIGURES_DIR / "phase4_logistic_coefficients.png"
POSITIVE = "Placed"
NEGATIVE = "Not Placed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # Guard against accidentally rerunning the final test evaluation and overwriting it.
    if REPORT_JSON.exists():
        raise FileExistsError(
            f"Phase 4 report already exists: {REPORT_JSON}. Evaluation is designed to run once. "
            "Do not delete reports to rerun casually; review the existing result instead."
        )
    for path in (config.TEST_DATA_PATH, MODEL_PATH, META_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    if metadata.get("test_set_used_during_selection") is not False:
        raise ValueError("Model metadata does not certify that the test set was excluded from selection.")
    if metadata.get("selected_model") != "Logistic Regression":
        raise ValueError(f"Expected Phase 3 Logistic Regression selection, got {metadata.get('selected_model')!r}")
    if metadata.get("model_features") != config.MODEL_FEATURES:
        raise ValueError("Model metadata features differ from current config.MODEL_FEATURES.")

    before_hash = sha256(config.TEST_DATA_PATH)
    test = pd.read_csv(config.TEST_DATA_PATH)
    expected_columns = config.MODEL_FEATURES + [config.TARGET_COLUMN]
    missing = sorted(set(expected_columns) - set(test.columns))
    if missing:
        raise ValueError(f"Test CSV missing required columns: {missing}")
    if test.empty or test[config.TARGET_COLUMN].isna().any():
        raise ValueError("Test data is empty or target contains missing values.")
    if set(test[config.TARGET_COLUMN].unique()) != {NEGATIVE, POSITIVE}:
        raise ValueError("Unexpected target labels in test CSV.")
    forbidden = set(config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS + config.EXCLUDED_FAIRNESS_COLUMNS)
    if forbidden.intersection(config.MODEL_FEATURES):
        raise ValueError("Forbidden columns are present in model feature list.")

    # Load the saved pipeline only; there is intentionally no .fit(), CV, or tuning call here.
    model = joblib.load(MODEL_PATH)
    X_test = test[config.MODEL_FEATURES].copy()
    y_test = test[config.TARGET_COLUMN].copy()
    # One held-out evaluation pass: labels and class probabilities from the saved model.
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.named_steps["model"].classes_)
    if POSITIVE not in classes:
        raise ValueError(f"Positive class {POSITIVE!r} not found in model classes {classes}")
    positive_index = classes.index(POSITIVE)
    positive_probability = proba[:, positive_index]

    cm = confusion_matrix(y_test, y_pred, labels=[NEGATIVE, POSITIVE])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 4 - one-time held-out test evaluation",
        "model": metadata["selected_model"],
        "model_artifact": str(MODEL_PATH.relative_to(config.PROJECT_ROOT)),
        "test_rows": int(len(test)),
        "training_rows_used_to_fit_selected_model": int(metadata["training_rows"]),
        "feature_count": len(config.MODEL_FEATURES),
        "positive_class": POSITIVE,
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_placed": float(precision_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0)),
        "recall_placed": float(recall_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0)),
        "f1_placed": float(f1_score(y_test, y_pred, pos_label=POSITIVE, zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == POSITIVE).astype(int), positive_probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix_labels": [NEGATIVE, POSITIVE],
        "confusion_matrix_rows_actual_cols_predicted": [[tn, fp], [fn, tp]],
        "tn_not_placed_correct": tn,
        "fp_false_placed_prediction": fp,
        "fn_missed_placed_student": fn,
        "tp_placed_correct": tp,
        "test_target_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "predicted_target_counts": {str(k): int(v) for k, v in pd.Series(y_pred).value_counts().items()},
        "test_set_sha256_before": before_hash,
        "test_set_sha256_after": sha256(config.TEST_DATA_PATH),
        "test_file_unchanged": before_hash == sha256(config.TEST_DATA_PATH),
        "test_set_used_for_training_or_selection": False,
        "fit_or_tuning_performed_in_phase4": False,
        "features": config.MODEL_FEATURES,
        "excluded_from_model_features": config.MODEL_EXCLUDED_COLUMNS,
        "sklearn_version": sklearn.__version__,
        "classification_report": classification_report(y_test, y_pred, labels=[NEGATIVE, POSITIVE], output_dict=True, zero_division=0),
        "limitations": [
            "This is a single evaluation of one previously selected model on one held-out split; it is not a guarantee of future performance.",
            "The dataset's real-world provenance and representativeness have not been independently established.",
            "Excluding age and gender does not guarantee fairness because other predictors may be proxies.",
            "The target is historical placement status; model associations do not establish causation.",
            "Predicted probabilities may not be calibrated and must not be presented as guaranteed individual outcomes.",
        ],
    }
    if not metrics["test_file_unchanged"]:
        raise RuntimeError("Test CSV changed during evaluation; investigate before trusting this report.")

    REPORT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=[f"Actual {NEGATIVE}", f"Actual {POSITIVE}"], columns=[f"Predicted {NEGATIVE}", f"Predicted {POSITIVE}"]).to_csv(CONFUSION_CSV, index=True)
    pred_df = pd.DataFrame({
        "actual_placement_status": y_test.to_numpy(),
        "predicted_placement_status": y_pred,
        "predicted_probability_placed": positive_probability,
    })
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[NEGATIVE, POSITIVE])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Phase 4: Held-out Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=160)
    plt.close(fig)

    # Interpretable logistic-regression coefficients, ordered by absolute magnitude.
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = np.asarray(classifier.coef_).ravel()
    coef_df = pd.DataFrame({"transformed_feature": names, "coefficient": coefs, "absolute_coefficient": np.abs(coefs)})
    coef_df = coef_df.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(COEFFICIENTS_CSV, index=False)
    top = coef_df.head(20).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["transformed_feature"], top["coefficient"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Logistic regression coefficient (transformed feature scale)")
    ax.set_title("Top 20 Logistic Regression Coefficients by Absolute Magnitude")
    fig.tight_layout()
    fig.savefig(COEFFICIENTS_PNG, dpi=160)
    plt.close(fig)

    report_lines = [
        "# Phase 4 — Held-out Test Evaluation", "",
        "## Evaluation protocol", "",
        f"- Evaluated the already-selected `{metadata['selected_model']}` pipeline once on `data/processed/test.csv` ({len(test):,} rows).",
        "- Loaded the saved Phase 3 artifact; no fitting, model selection, feature selection, or tuning occurred in this phase.",
        "- Test set was not used in Phase 3 model selection. The test CSV SHA-256 matched before and after evaluation.",
        "- Positive class: `Placed`; classification threshold: 0.5.",
        "- Excluded model inputs: `student_id`, `salary_package_lpa`, `age`, `gender`.", "",
        "## Final test metrics", "",
        "| Metric | Test result |", "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |",
        f"| Precision (Placed) | {metrics['precision_placed']:.4f} |",
        f"| Recall (Placed) | {metrics['recall_placed']:.4f} |",
        f"| F1 (Placed) | {metrics['f1_placed']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
        f"| Matthews correlation coefficient | {metrics['matthews_corrcoef']:.4f} |", "",
        "## Confusion matrix", "",
        "Rows are actual labels; columns are predicted labels.", "",
        f"| Actual \\ Predicted | Not Placed | Placed |", "|---|---:|---:|",
        f"| Not Placed | TN = {tn} | FP = {fp} |",
        f"| Placed | FN = {fn} | TP = {tp} |", "",
        f"- False positives: {fp} students predicted as placed who were not placed in the dataset.",
        f"- False negatives: {fn} students predicted as not placed who were placed in the dataset.", "",
        "## Coefficient-based feature interpretation", "",
        "The coefficient file ranks transformed features by absolute logistic-regression coefficient. Positive coefficients are associated with higher model log-odds of `Placed`; negative coefficients with lower log-odds, conditional on the model's other inputs. Numeric features were standardized and categorical features one-hot encoded.",
        "These are model associations, not causal effects. Coefficient magnitude depends on encoding and scaling; correlated predictors can share or distort apparent importance. Do not treat this ranking as proof that changing a feature will change placement outcomes.", "",
        "### Top 10 by absolute coefficient", "",
        "| Transformed feature | Coefficient |", "|---|---:|",
    ]
    for _, row in coef_df.head(10).iterrows():
        report_lines.append(f"| `{row['transformed_feature']}` | {row['coefficient']:.5f} |")
    report_lines += ["", "## Limitations and responsible use", ""]
    report_lines += [f"- {item}" for item in metrics["limitations"]]
    report_lines += [
        "- This is a baseline classifier, not a hiring or admissions decision system. Avoid using it to make consequential decisions about individual students.",
        "- Before real deployment, validate provenance, leakage, representativeness, calibration, subgroup performance, and prospective performance on independently collected data.",
        "", "## Artifacts", "",
        "- `reports/phase4_test_metrics.json` — metrics, target counts, hashes, protocol metadata.",
        "- `reports/phase4_test_predictions.csv` — held-out predictions and predicted probabilities.",
        "- `reports/phase4_confusion_matrix.csv` and `reports/figures/phase4_confusion_matrix.png`.",
        "- `reports/phase4_logistic_coefficients.csv` and `reports/figures/phase4_logistic_coefficients.png`.",
        "", "## Automated validation", "",
        f"- Test rows: {len(test):,}; prediction count: {len(y_pred):,}.",
        f"- Probability array shape: {list(proba.shape)}; probability values finite: {bool(np.isfinite(proba).all())}.",
        f"- Confusion-matrix total equals test rows: {int(cm.sum()) == len(test)}.",
        f"- Metrics finite: {all(np.isfinite(metrics[k]) for k in ['accuracy','precision_placed','recall_placed','f1_placed','roc_auc'])}.",
        f"- Test file unchanged: {metrics['test_file_unchanged']}.",
        "- Model fit/tuning in Phase 4: no.", "",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    # Validate saved artifacts without re-running model inference.
    saved = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    pred_saved = pd.read_csv(PREDICTIONS_PATH)
    cm_saved = pd.read_csv(CONFUSION_CSV, index_col=0)
    assert len(pred_saved) == len(test)
    assert int(cm_saved.to_numpy().sum()) == len(test)
    assert saved["test_file_unchanged"] is True
    assert saved["test_set_used_for_training_or_selection"] is False
    assert saved["fit_or_tuning_performed_in_phase4"] is False
    assert np.isfinite(pred_saved["predicted_probability_placed"]).all()
    assert pred_saved["predicted_probability_placed"].between(0, 1).all()

    print("Phase 4 completed and validated.")
    print(f"Test rows: {len(test):,}")
    for key in ("accuracy", "precision_placed", "recall_placed", "f1_placed", "roc_auc", "balanced_accuracy", "matthews_corrcoef"):
        print(f"{key}: {metrics[key]:.6f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {[[tn, fp], [fn, tp]]}")
    print(f"Test file unchanged: {metrics['test_file_unchanged']}")
    print(f"Report: {REPORT_MD}")

if __name__ == "__main__":
    main()
