"""Local pre-run checks for the Student Placement Prediction project.

This script validates project files and a single inference smoke test. It does not
fit/tune a model and does not calculate any metrics on the held-out test set.
"""
from __future__ import annotations
import hashlib
import json
import py_compile
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

MODEL_PATH = ROOT / "models" / "selected_placement_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
TRAIN_PATH = config.TRAIN_DATA_PATH
TEST_PATH = config.TEST_DATA_PATH
RAW_PATH = ROOT / "data" / "raw" / "student_placement_prediction_dataset_2026.csv"

checks: list[dict[str, object]] = []
def record(name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})

for rel in ["app.py", "phase1_audit.py", "phase2_split_eda.py", "phase3_train_select.py", "phase4_evaluate.py", "phase5a_diagnose_compare.py", "phase5b_feature_engineering.py"]:
    path = ROOT / rel
    try:
        py_compile.compile(str(path), doraise=True)
        record(f"compile:{rel}", True, "Python syntax compiled")
    except Exception as exc:
        record(f"compile:{rel}", False, str(exc))

for path, name in [(RAW_PATH, "raw dataset"), (TRAIN_PATH, "training split"), (TEST_PATH, "test split"), (MODEL_PATH, "selected model"), (METADATA_PATH, "model metadata")]:
    record(f"exists:{name}", path.exists(), str(path.relative_to(ROOT)))

if RAW_PATH.exists() and TRAIN_PATH.exists() and TEST_PATH.exists():
    raw = pd.read_csv(RAW_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    record("dataset row counts", len(raw) == 100000 and len(train) == 80000 and len(test) == 20000,
           f"raw={len(raw)}, train={len(train)}, test={len(test)}")
    expected_split_columns = [c for c in raw.columns if c not in (config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS)]
    record("train/test columns align", list(train.columns) == list(test.columns) == expected_split_columns,
           "Train/test columns match raw schema order after intentionally excluding identifier and post-outcome salary")
    record("target labels valid", set(train[config.TARGET_COLUMN].dropna().unique()) <= set(config.TARGET_MAPPING) and set(test[config.TARGET_COLUMN].dropna().unique()) <= set(config.TARGET_MAPPING),
           "Placement target contains only configured labels")

if MODEL_PATH.exists() and METADATA_PATH.exists() and TRAIN_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    expected = metadata.get("model_features", [])
    forbidden = {"student_id", "salary_package_lpa", "age", "gender", config.TARGET_COLUMN}
    record("feature policy", bool(expected) and not (set(expected) & forbidden),
           f"features={len(expected)}; forbidden inputs absent={not (set(expected) & forbidden)}")
    train = pd.read_csv(TRAIN_PATH, usecols=expected)
    sample = train.iloc[[0]].copy()
    try:
        pred = model.predict(sample)
        proba = model.predict_proba(sample)
        valid = len(pred) == 1 and proba.shape == (1, 2) and pd.notna(proba).all() and ((proba >= 0) & (proba <= 1)).all() and abs(float(proba.sum()) - 1.0) < 1e-8
        record("saved model inference smoke test", bool(valid), f"prediction={str(pred[0])}; probabilities={proba.tolist()}")
    except Exception as exc:
        record("saved model inference smoke test", False, str(exc))

result = {
    "project": "Student Placement Prediction System",
    "validation_type": "file, syntax, schema, and one-row training-data inference smoke checks",
    "held_out_test_metrics_calculated": False,
    "model_fitted_or_tuned": False,
    "checks_passed": sum(bool(c["passed"]) for c in checks),
    "checks_total": len(checks),
    "passed": all(bool(c["passed"]) for c in checks),
    "checks": checks,
}
out = ROOT / "reports" / "final_project_validation.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
