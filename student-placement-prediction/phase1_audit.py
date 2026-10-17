"""Audit the raw CSV and write a reproducible Phase 1 JSON report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config


def build_audit(data_path: Path = config.SOURCE_DATA_PATH) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    numeric = df.select_dtypes(include="number")
    target_counts = df[config.TARGET_COLUMN].value_counts(dropna=False).to_dict()
    salary_by_target = {}
    if config.POST_OUTCOME_COLUMNS[0] in df.columns:
        salary_by_target = {
            str(label): {
                "min": float(group.min()),
                "max": float(group.max()),
                "count": int(group.count()),
                "nunique": int(group.nunique()),
            }
            for label, group in df.groupby(config.TARGET_COLUMN)[config.POST_OUTCOME_COLUMNS[0]]
        }
    return {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(data_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "expected_shape": {"rows": config.EXPECTED_SHAPE[0], "columns": config.EXPECTED_SHAPE[1]},
        "shape_matches_specification": tuple(df.shape) == config.EXPECTED_SHAPE,
        "columns": list(df.columns),
        "expected_columns": config.EXPECTED_COLUMNS,
        "columns_match_specification_ordered": list(df.columns) == config.EXPECTED_COLUMNS,
        "missing_expected_columns": sorted(set(config.EXPECTED_COLUMNS) - set(df.columns)),
        "unexpected_columns": sorted(set(df.columns) - set(config.EXPECTED_COLUMNS)),
        "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
        "missing_values_by_column": {c: int(n) for c, n in df.isna().sum().items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_student_ids": int(df["student_id"].duplicated().sum()) if "student_id" in df else None,
        "target_column": config.TARGET_COLUMN,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_percentages": {str(k): float(v / len(df) * 100) for k, v in target_counts.items()},
        "numeric_summary": {
            c: {k: float(v) for k, v in numeric[c].describe().items()}
            for c in numeric.columns
        },
        "salary_by_target": salary_by_target,
        "modeling_policy": {
            "excluded_identifier_columns": config.ID_COLUMNS,
            "excluded_post_outcome_columns": config.POST_OUTCOME_COLUMNS,
            "excluded_fairness_columns": config.EXCLUDED_FAIRNESS_COLUMNS,
            "target_mapping": config.TARGET_MAPPING,
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "model_features": config.MODEL_FEATURES,
        },
    }


def main() -> None:
    audit = build_audit()
    config.PHASE1_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PHASE1_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Audit written: {config.PHASE1_AUDIT_PATH}")
    print(f"Shape: {audit['shape']['rows']:,} rows x {audit['shape']['columns']} columns")
    print(f"Missing values: {audit['missing_values_total']:,}; duplicate rows: {audit['duplicate_rows']:,}")
    print(f"Target counts: {audit['target_counts']}")
    print(f"Specification shape match: {audit['shape_matches_specification']}")
    print(f"Specification column order match: {audit['columns_match_specification_ordered']}")


if __name__ == "__main__":
    main()
