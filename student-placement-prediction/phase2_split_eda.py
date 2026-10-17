"""Validate Phase 1 artifacts, create a stratified split, and run training-only EDA."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from phase1_audit import build_audit


LABEL_ORDER = ["Not Placed", "Placed"]


def validate_phase1_artifacts(df: pd.DataFrame) -> dict:
    """Check config, JSON audit, and current raw data agree before Phase 2 proceeds."""
    if not config.PHASE1_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Phase 1 audit JSON is missing: {config.PHASE1_AUDIT_PATH}. Run phase1_audit.py first."
        )
    saved = json.loads(config.PHASE1_AUDIT_PATH.read_text(encoding="utf-8"))
    actual = build_audit(config.SOURCE_DATA_PATH)
    checks = {
        "actual_shape_matches_config": tuple(df.shape) == config.EXPECTED_SHAPE,
        "actual_columns_match_config": list(df.columns) == config.EXPECTED_COLUMNS,
        "saved_audit_shape_matches_actual": saved.get("shape") == actual.get("shape"),
        "saved_audit_columns_match_actual": saved.get("columns") == actual.get("columns"),
        "saved_audit_dtypes_match_actual": saved.get("dtypes") == actual.get("dtypes"),
        "saved_audit_missing_counts_match_actual": saved.get("missing_values_by_column") == actual.get("missing_values_by_column"),
        "saved_audit_target_counts_match_actual": saved.get("target_counts") == actual.get("target_counts"),
        "saved_audit_expected_columns_match_config": saved.get("expected_columns") == config.EXPECTED_COLUMNS,
        "saved_audit_policy_matches_config": saved.get("modeling_policy") == actual.get("modeling_policy"),
        "saved_audit_specification_flags_match_actual": (
            saved.get("shape_matches_specification") == actual.get("shape_matches_specification")
            and saved.get("columns_match_specification_ordered") == actual.get("columns_match_specification_ordered")
        ),
        "saved_audit_duplicate_counts_match_actual": (
            saved.get("duplicate_rows") == actual.get("duplicate_rows")
            and saved.get("duplicate_student_ids") == actual.get("duplicate_student_ids")
        ),
        "target_mapping_covers_actual_labels": set(df[config.TARGET_COLUMN].dropna().unique()) == set(config.TARGET_MAPPING),
        "id_excluded_from_model_features": not set(config.ID_COLUMNS).intersection(config.MODEL_FEATURES),
        "salary_excluded_from_model_features": not set(config.POST_OUTCOME_COLUMNS).intersection(config.MODEL_FEATURES),
        "age_gender_excluded_from_model_features": not set(config.EXCLUDED_FAIRNESS_COLUMNS).intersection(config.MODEL_FEATURES),
        "target_excluded_from_model_features": config.TARGET_COLUMN not in config.MODEL_FEATURES,
        "configured_features_exist": set(config.MODEL_FEATURES).issubset(df.columns),
    }
    failed = [name for name, ok in checks.items() if not ok]
    result = {"checks": checks, "passed": not failed, "failed_checks": failed}
    if failed:
        raise ValueError("Phase 1 artifact consistency checks failed: " + ", ".join(failed))
    return result


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_target_distribution(train: pd.DataFrame) -> dict:
    counts = train[config.TARGET_COLUMN].value_counts().reindex(LABEL_ORDER, fill_value=0)
    percentages = counts / len(train) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values)
    ax.set_title("Training-set placement status distribution")
    ax.set_ylabel("Number of students")
    ax.bar_label(bars, labels=[f"{v:,}\n({p:.2f}%)" for v, p in zip(counts.values, percentages.values)])
    save_figure(config.FIGURES_DIR / "target_distribution_train.png")
    return {"counts": {k: int(v) for k, v in counts.items()}, "percentages": {k: float(v) for k, v in percentages.items()}}


def make_numeric_plots(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [c for c in config.EDA_NUMERICAL_COLUMNS if c in train.columns]
    grouped = train.groupby(config.TARGET_COLUMN)[columns].agg(["mean", "median", "std"])
    grouped.to_csv(config.REPORTS_DIR / "numeric_classwise_summary.csv")

    # Distribution histograms by class: one file per feature for legible comparisons.
    for col in columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label in LABEL_ORDER:
            values = train.loc[train[config.TARGET_COLUMN] == label, col].dropna()
            ax.hist(values, bins=30, alpha=0.48, density=True, label=label)
        ax.set_title(f"{col}: training distribution by placement status")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        save_figure(config.FIGURES_DIR / f"distribution_{col}.png")

    # Compact boxplots for class-wise comparisons across numeric variables.
    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, columns):
        series = [train.loc[train[config.TARGET_COLUMN] == label, col].dropna() for label in LABEL_ORDER]
        ax.boxplot(series, tick_labels=LABEL_ORDER, showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", labelrotation=12)
    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle("Numeric feature comparison by placement status (training set)", y=1.002)
    save_figure(config.FIGURES_DIR / "numeric_boxplots_by_target.png")

    # Numeric features' Pearson correlation with the binary target and feature-feature correlations.
    target_encoded = train[config.TARGET_COLUMN].map(config.TARGET_MAPPING).astype(float)
    numeric_frame = train[columns].copy()
    numeric_frame["placement_status_encoded"] = target_encoded
    corr = numeric_frame.corr(method="pearson")
    corr.to_csv(config.REPORTS_DIR / "numeric_correlation_matrix_train.csv")
    numeric_frame.corr(method="spearman").to_csv(config.REPORTS_DIR / "numeric_spearman_correlation_matrix_train.csv")
    target_corr = corr["placement_status_encoded"].drop("placement_status_encoded").sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr.rename("pearson_correlation_with_placed_1").to_csv(config.REPORTS_DIR / "numeric_target_correlations_train.csv", header=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    ax.set_title("Pearson correlations: numeric features and encoded target (training set)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Pearson r")
    save_figure(config.FIGURES_DIR / "numeric_correlation_heatmap_train.png")
    return grouped, target_corr.to_frame()


def make_categorical_analysis(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.EDA_CATEGORICAL_COLUMNS:
        table = pd.crosstab(train[col], train[config.TARGET_COLUMN]).reindex(columns=LABEL_ORDER, fill_value=0)
        table["total"] = table.sum(axis=1)
        table["placed_rate_pct"] = table["Placed"] / table["total"] * 100
        table["not_placed_rate_pct"] = table["Not Placed"] / table["total"] * 100
        table.insert(0, "feature", col)
        table.insert(1, "category", table.index.astype(str))
        rows.extend(table.reset_index(drop=True).to_dict(orient="records"))

        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_table = table.sort_values("placed_rate_pct", ascending=False)
        bars = ax.bar(plot_table.index.astype(str), plot_table["placed_rate_pct"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Placed students (%)")
        ax.set_xlabel(col)
        ax.set_title(f"Observed placement rate by {col} (training set)")
        ax.tick_params(axis="x", labelrotation=20)
        ax.bar_label(bars, labels=[f"{x:.1f}%\nn={n:,}" for x, n in zip(plot_table["placed_rate_pct"], plot_table["total"])], fontsize=8)
        save_figure(config.FIGURES_DIR / f"placement_rate_by_{col}.png")
    result = pd.DataFrame(rows)
    result.to_csv(config.REPORTS_DIR / "categorical_placement_rates_train.csv", index=False)
    return result


def write_findings(train: pd.DataFrame, test: pd.DataFrame, checks: dict, target_dist: dict,
                   target_corr: pd.DataFrame, categorical: pd.DataFrame) -> dict:
    counts = train[config.TARGET_COLUMN].value_counts()
    class_means = train.groupby(config.TARGET_COLUMN)[config.EDA_NUMERICAL_COLUMNS].mean()
    numeric_corr = target_corr.iloc[:, 0]
    strongest = numeric_corr.reindex(numeric_corr.abs().sort_values(ascending=False).head(8).index)
    cat_summary = []
    for col in config.EDA_CATEGORICAL_COLUMNS:
        subset = categorical[categorical["feature"] == col].sort_values("placed_rate_pct", ascending=False)
        cat_summary.append({
            "feature": col,
            "highest_observed_placement_rate_category": str(subset.iloc[0]["category"]),
            "highest_rate_pct": float(subset.iloc[0]["placed_rate_pct"]),
            "lowest_observed_placement_rate_category": str(subset.iloc[-1]["category"]),
            "lowest_rate_pct": float(subset.iloc[-1]["placed_rate_pct"]),
            "note": "Descriptive association only; category sizes and confounding matter.",
        })
    age_means = {str(k): float(v) for k, v in train.groupby(config.TARGET_COLUMN)["age"].mean().items()}
    gender_rates = (train.groupby("gender")[config.TARGET_COLUMN].apply(lambda s: (s == "Placed").mean() * 100)).to_dict()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(config.SOURCE_DATA_PATH),
        "eda_data_scope": "training set only; test split was not used for analysis",
        "split": {
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE,
            "train_target_counts": {str(k): int(v) for k, v in counts.items()},
            "train_target_percentages": target_dist["percentages"],
            "test_target_counts_saved_for_split_validation_only": {str(k): int(v) for k, v in test[config.TARGET_COLUMN].value_counts().items()},
        },
        "phase1_artifact_consistency": checks,
        "target_correlations_numeric_pearson": {str(k): float(v) for k, v in numeric_corr.items()},
        "strongest_absolute_numeric_target_correlations_top8": {str(k): float(v) for k, v in strongest.items()},
        "categorical_placement_rate_extremes": cat_summary,
        "age_decision": {
            "decision": "exclude from model features; retain for training-only EDA and potential audit",
            "reason": "age spans a narrow 18-24 range and class-wise averages are nearly identical; age is also demographic and offers little demonstrated incremental value at this stage",
            "training_mean_age_by_target": age_means,
        },
        "gender_decision": {
            "decision": "exclude from model features; retain for descriptive fairness audit only",
            "reason": "gender is a sensitive demographic attribute; excluding it is a conservative baseline choice, though other features can still act as proxies and exclusion alone does not guarantee fairness",
            "training_observed_placement_rate_pct_by_gender": {str(k): float(v) for k, v in gender_rates.items()},
        },
        "model_feature_policy": {
            "feature_count": len(config.MODEL_FEATURES),
            "features": config.MODEL_FEATURES,
            "excluded_columns": config.MODEL_EXCLUDED_COLUMNS + [config.TARGET_COLUMN],
            "app_input_policy": "Streamlit form must request exactly MODEL_FEATURES; no default age or gender values because neither is a model input.",
        },
        "interpretation_limits": [
            "EDA relationships are descriptive and do not establish causation.",
            "A random split measures performance on a similar data distribution, not necessarily on future cohorts or other colleges.",
            "Dataset provenance and possible synthetic-data generation should be verified before making real-world claims.",
        ],
    }
    config.EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 2 — Train-Test Split and Exploratory Data Analysis",
        "",
        "## Scope and reproducibility",
        f"- Source CSV: `{config.SOURCE_DATA_PATH}` (left unchanged).",
        f"- Split: stratified {int((1-config.TEST_SIZE)*100)}:{int(config.TEST_SIZE*100)}, `random_state={config.RANDOM_STATE}`.",
        f"- Training rows: {len(train):,}; test rows: {len(test):,}.",
        "- EDA charts and association summaries were computed from the training split only. The test split was not used to select features or inspect feature relationships.",
        "- The saved split files omit `student_id` and `salary_package_lpa`; age and gender remain in the split files for descriptive auditing, but are excluded from model features.",
        "",
        "## Target distribution in training data",
    ]
    for label, value in target_dist["counts"].items():
        lines.append(f"- {label}: {value:,} ({target_dist['percentages'][label]:.2f}%).")
    lines += ["", "## Numeric features with strongest absolute Pearson association with target", "", "These are univariate correlations with `Placed=1` and `Not Placed=0`; they are not causal effects and do not capture nonlinear relationships or interactions.", ""]
    for feature, value in strongest.items():
        lines.append(f"- `{feature}`: r = {value:.4f}.")
    lines += ["", "## Categorical placement-rate patterns", "", "Observed rates are descriptive and can reflect confounding or different group sizes.", ""]
    for item in cat_summary:
        lines.append(f"- `{item['feature']}`: highest observed rate `{item['highest_observed_placement_rate_category']}` ({item['highest_rate_pct']:.2f}%); lowest `{item['lowest_observed_placement_rate_category']}` ({item['lowest_rate_pct']:.2f}%).")
    lines += ["", "## Age and gender policy", "", f"- **Age:** excluded from model features. Training-set mean age by target: {age_means}. It has a narrow range and little apparent univariate separation; retain only for descriptive checks unless a defensible use case is established.", f"- **Gender:** excluded from model features as a conservative fairness choice. Observed training placement rates by gender: { {k: round(v, 2) for k, v in gender_rates.items()} }. These rates should be reviewed for fairness, not treated as causal differences.", "- The future application must request exactly the configured model features. It must not silently fill age/gender with arbitrary defaults.", "", "## Leakage and quality safeguards", "", "- `student_id` is omitted from split files because it is an identifier.", "- `salary_package_lpa` is omitted because it is known after placement and directly reveals the target in this dataset.", "- Split is stratified; train/test class counts are recorded in the JSON summary for verification.", "- No model has been trained in this phase.", "", "## Generated artifacts", "", "- `data/processed/train.csv`, `data/processed/test.csv`", "- `reports/phase1_audit.json`, `reports/phase2_eda_summary.json`", "- `reports/numeric_classwise_summary.csv`, `reports/numeric_correlation_matrix_train.csv`, `reports/numeric_spearman_correlation_matrix_train.csv`, `reports/numeric_target_correlations_train.csv`, `reports/categorical_placement_rates_train.csv`", "- Charts under `reports/figures/`.", "", "## Limitations", "", "The dataset's provenance is not established by this analysis. Its size alone does not prove it represents real placement outcomes. Correlation and category-level rates do not establish causation, and a random split cannot guarantee generalization to future student cohorts.", ""]
    config.EDA_FINDINGS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    df = pd.read_csv(config.SOURCE_DATA_PATH)
    checks = validate_phase1_artifacts(df)
    print("Phase 1 consistency checks: PASS")
    for name, passed in checks["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    # Split from raw data, but do not carry the identifier or post-placement salary into either split.
    split_df = df.drop(columns=config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS).copy()
    train, test = train_test_split(
        split_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=split_df[config.TARGET_COLUMN],
    )
    # Validate the in-memory partition before discarding original row indices.
    assert set(train.index).isdisjoint(set(test.index)), "Train/test original row indices overlap"
    assert len(train) + len(test) == len(split_df), "Split row counts do not sum to the source row count"
    assert set(train.index).union(set(test.index)) == set(split_df.index), "Some source rows are missing from the split"
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    config.TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_DATA_PATH, index=False)
    test.to_csv(config.TEST_DATA_PATH, index=False)

    # Re-read saved outputs and validate their sizes, schema, disjointness, and class proportions.
    train_saved = pd.read_csv(config.TRAIN_DATA_PATH)
    test_saved = pd.read_csv(config.TEST_DATA_PATH)
    expected_split_columns = [c for c in config.EXPECTED_COLUMNS if c not in config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS]
    assert len(train_saved) == 80_000, f"Expected 80000 train rows, got {len(train_saved)}"
    assert len(test_saved) == 20_000, f"Expected 20000 test rows, got {len(test_saved)}"
    assert list(train_saved.columns) == expected_split_columns
    assert list(test_saved.columns) == expected_split_columns
    assert not set(config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS).intersection(train_saved.columns)
    assert not set(config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS).intersection(test_saved.columns)
    assert set(train_saved[config.TARGET_COLUMN].unique()) == set(config.TARGET_MAPPING)
    assert set(test_saved[config.TARGET_COLUMN].unique()) == set(config.TARGET_MAPPING)
    assert train_saved[config.TARGET_COLUMN].value_counts().to_dict() == train[config.TARGET_COLUMN].value_counts().to_dict()

    target_dist = make_target_distribution(train_saved)
    _, target_corr = make_numeric_plots(train_saved)
    categorical = make_categorical_analysis(train_saved)
    summary = write_findings(train_saved, test_saved, checks, target_dist, target_corr, categorical)

    print(f"Saved train: {config.TRAIN_DATA_PATH} ({len(train_saved):,} rows x {train_saved.shape[1]} columns)")
    print(f"Saved test:  {config.TEST_DATA_PATH} ({len(test_saved):,} rows x {test_saved.shape[1]} columns)")
    print(f"EDA findings: {config.EDA_FINDINGS_PATH}")
    print(f"EDA summary: {config.EDA_SUMMARY_PATH}")
    print(f"Figures created: {len(list(config.FIGURES_DIR.glob('*.png')))}")
    print("Phase 2 completed. No model was trained.")
    print("Top numeric correlations (absolute):")
    for feature, value in summary["strongest_absolute_numeric_target_correlations_top8"].items():
        print(f"  {feature}: {value:+.4f}")


if __name__ == "__main__":
    main()
