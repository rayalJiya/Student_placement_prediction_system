"""Phase 5B: training-only feature engineering screen. Never opens test.csv."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import config

CV_SAMPLE_SIZE = 24000
N_SPLITS = 3
TARGET = config.TARGET_COLUMN
POSITIVE = "Placed"
ENGINEERED = [
    "technical_academic_mean",
    "experience_activity_count",
    "career_presence_mean_log1p",
    "interview_readiness_mean",
    "leadership_engagement_mean",
]

class PlacementFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add transparent, pre-outcome summary features; preserve all original inputs."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=config.MODEL_FEATURES)
        out = X.copy()
        def mean_of(cols):
            return out[cols].astype(float).mean(axis=1)
        out["technical_academic_mean"] = mean_of([
            "cgpa", "coding_skill_score", "aptitude_score", "logical_reasoning_score"
        ])
        out["experience_activity_count"] = out[[
            "internships_count", "projects_count", "certifications_count", "hackathons_participated"
        ]].astype(float).sum(axis=1)
        out["career_presence_mean_log1p"] = np.log1p(out[[
            "github_repos", "linkedin_connections"
        ]].astype(float)).mean(axis=1)
        out["interview_readiness_mean"] = mean_of([
            "coding_skill_score", "aptitude_score", "communication_skill_score", "mock_interview_score"
        ])
        volunteer = out["volunteer_experience"].map({"Yes": 1.0, "No": 0.0}).fillna(0.0)
        out["leadership_engagement_mean"] = pd.concat([
            out["extracurricular_score"].astype(float),
            out["leadership_score"].astype(float), volunteer.rename("volunteer_binary")
        ], axis=1).mean(axis=1)
        return out

def make_preprocessor(numeric_features, dense=False, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", cat_pipe, config.CATEGORICAL_FEATURES),
    ], sparse_threshold=0.0 if dense else 0.3)

def build_candidates():
    base_pre = make_preprocessor(config.NUMERICAL_FEATURES, scale=True)
    eng_num = config.NUMERICAL_FEATURES + ENGINEERED
    eng_pre = make_preprocessor(eng_num, scale=True)
    return {
        "Logistic Regression baseline": Pipeline([
            ("preprocess", base_pre),
            ("model", LogisticRegression(max_iter=1500, random_state=config.RANDOM_STATE)),
        ]),
        "Logistic Regression + engineered summaries": Pipeline([
            ("feature_engineering", PlacementFeatureEngineer()),
            ("preprocess", eng_pre),
            ("model", LogisticRegression(max_iter=1500, random_state=config.RANDOM_STATE)),
        ]),
        "HistGradientBoosting + engineered summaries": Pipeline([
            ("feature_engineering", PlacementFeatureEngineer()),
            ("preprocess", make_preprocessor(eng_num, dense=True, scale=False)),
            ("model", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08,
                max_leaf_nodes=15, l2_regularization=1.0, random_state=config.RANDOM_STATE)),
        ]),
    }

def main():
    started = time.time()
    out_dir = config.REPORTS_DIR
    if not config.TRAIN_DATA_PATH.exists():
        raise FileNotFoundError("Training split missing. Run Phase 2 first.")
    # Deliberately open training split only. Do not read test.csv or previous test predictions.
    train = pd.read_csv(config.TRAIN_DATA_PATH)
    needed = set(config.MODEL_FEATURES + [TARGET])
    missing = sorted(needed - set(train.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")
    forbidden = set(config.ID_COLUMNS + config.POST_OUTCOME_COLUMNS + config.EXCLUDED_FAIRNESS_COLUMNS)
    if forbidden.intersection(config.MODEL_FEATURES):
        raise ValueError("Excluded columns found in model features")
    X, y = train[config.MODEL_FEATURES].copy(), train[TARGET].copy()
    if len(X) > CV_SAMPLE_SIZE:
        X_cv, _, y_cv, _ = train_test_split(X, y, train_size=CV_SAMPLE_SIZE,
            stratify=y, random_state=config.RANDOM_STATE)
    else:
        X_cv, y_cv = X, y
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    scorers = {
        "accuracy": "accuracy",
        "precision_placed": make_scorer(precision_score, pos_label=POSITIVE, zero_division=0),
        "recall_placed": make_scorer(recall_score, pos_label=POSITIVE, zero_division=0),
        "f1_placed": make_scorer(f1_score, pos_label=POSITIVE, zero_division=0),
        "roc_auc": "roc_auc",
    }
    results=[]
    for name, pipe in build_candidates().items():
        print(f"CV: {name}", flush=True)
        t0=time.time()
        scores=cross_validate(pipe, X_cv, y_cv, cv=cv, scoring=scorers,
            n_jobs=1, return_train_score=False, error_score="raise")
        row={"model":name,"cv_rows":len(X_cv),"cv_folds":N_SPLITS,
             "fit_seconds_total":round(float(scores["fit_time"].sum()),3)}
        for metric in scorers:
            vals=scores[f"test_{metric}"]
            row[f"{metric}_mean"]=float(np.mean(vals))
            row[f"{metric}_std"]=float(np.std(vals,ddof=1))
        results.append(row)
        print(f"  ROC-AUC={row['roc_auc_mean']:.4f}, F1={row['f1_placed_mean']:.4f}, elapsed={time.time()-t0:.1f}s",flush=True)
    result_df=pd.DataFrame(results).sort_values(["roc_auc_mean","f1_placed_mean"],ascending=False)
    result_path=out_dir/"phase5b_feature_engineering_cv.csv"
    result_df.to_csv(result_path,index=False)
    base=result_df[result_df.model=="Logistic Regression baseline"].iloc[0]
    eng=result_df[result_df.model=="Logistic Regression + engineered summaries"].iloc[0]
    best=result_df.iloc[0]
    provenance={
        "overview_mentions_source_or_generation_method": any(k in Path(config.PROJECT_ROOT.parent/"overview.txt").read_text(encoding="utf-8").lower() for k in ["generated by", "synthetic data", "data source", "dataset source", "generation process"]),
        "dataset_provenance_confirmed": False,
        "label_generation_rule_documented": False,
        "note": "Project overview defines the target and expected columns but does not establish the dataset publisher, collection process, or placement_status generation rule. These cannot be inferred reliably from observed correlations alone."
    }
    report=[
        "# Phase 5B — Training-only feature engineering and provenance review", "",
        "## Data protection", "",
        f"- Read only `data/processed/train.csv` ({len(train):,} rows). `test.csv` was not opened.",
        f"- Comparison used {len(X_cv):,} stratified training rows and {N_SPLITS}-fold StratifiedKFold (random_state={config.RANDOM_STATE}).",
        "- All feature engineering is inside each cross-validation pipeline, so summaries are recomputed from each fold's input rows.",
        "- No model was refit for deployment, no saved model was overwritten, and no test metrics were calculated.",
        "", "## Provenance and label-generation review", "",
        "- The project overview describes `placement_status` as the target and `salary_package_lpa` as a post-placement outcome.",
        "- The provided project files do not identify a trusted publisher, collection period, data dictionary beyond column descriptions, or the rule used to create placement labels.",
        "- Therefore dataset provenance and label-generation process remain **unverified**. Do not describe the dataset as real-world or synthetic without confirmation from its provider.",
        "- `salary_package_lpa` is excluded because it is known after the placement outcome and would leak outcome information.",
        "", "## Training-only comparison", "",
        "| Candidate | Accuracy | Precision (Placed) | Recall (Placed) | F1 (Placed) | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for _,r in result_df.iterrows():
        report.append(f"| {r['model']} | {r['accuracy_mean']:.4f} | {r['precision_placed_mean']:.4f} | {r['recall_placed_mean']:.4f} | {r['f1_placed_mean']:.4f} | {r['roc_auc_mean']:.4f} |")
    report += ["", "## Interpretation", "",
        f"- Highest mean ROC-AUC in this small screening run: **{best['model']}** (ROC-AUC={best['roc_auc_mean']:.4f}).",
        f"- Engineered Logistic Regression ROC-AUC change vs baseline: {eng['roc_auc_mean']-base['roc_auc_mean']:+.4f}; F1 change: {eng['f1_placed_mean']-base['f1_placed_mean']:+.4f}.",
        "- Engineered summaries are transparent combinations of existing pre-outcome fields; they do not add new information and may be redundant for linear models.",
        "- Small cross-validation differences on a 24,000-row subset are screening evidence only, not proof of a robust improvement.",
        "- No model change is recommended from this phase alone. Preserve the existing model and use the current results as research diagnostics.",
        "", "## Proposed engineered features", "",
        "- `technical_academic_mean`: mean of CGPA, coding, aptitude, and logical reasoning scores.",
        "- `experience_activity_count`: internships, projects, certifications, and hackathons summed.",
        "- `career_presence_mean_log1p`: mean log1p of GitHub repositories and LinkedIn connections.",
        "- `interview_readiness_mean`: mean of coding, aptitude, communication, and mock-interview scores.",
        "- `leadership_engagement_mean`: mean of extracurricular score, leadership score, and binary volunteer indicator.",
        "", "## Artifacts", "",
        "- `reports/phase5b_feature_engineering_cv.csv`",
        "- `reports/phase5b_provenance_review.json`",
    ]
    (out_dir/"phase5b_feature_engineering.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (out_dir/"phase5b_provenance_review.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    validation={"phase":"5B","training_rows":len(train),"cv_rows":len(X_cv),"cv_folds":N_SPLITS,
        "test_csv_opened":False,"test_set_used_for_tuning":False,"saved_model_overwritten":False,
        "results_rows":len(result_df),"all_metrics_finite":bool(np.isfinite(result_df.select_dtypes(include=[np.number]).to_numpy()).all()),
        "all_roc_auc_in_valid_range":bool(result_df.roc_auc_mean.between(0,1).all()),
        "engineered_features":ENGINEERED,"sklearn_version":sklearn.__version__,"elapsed_seconds":round(time.time()-started,2)}
    (out_dir/"phase5b_validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    print("\nPhase 5B complete. Best candidate:",best["model"],"ROC-AUC",round(float(best["roc_auc_mean"]),4),flush=True)
    print("Report:",out_dir/"phase5b_feature_engineering.md",flush=True)

if __name__=="__main__":
    main()
