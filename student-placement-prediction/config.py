"""Central paths and project settings for the Student Placement Prediction System."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SOURCE_DATA_PATH = PROJECT_ROOT.parent / "student_placement_prediction_dataset_2026.csv"
PHASE1_AUDIT_PATH = REPORTS_DIR / "phase1_audit.json"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
EDA_FINDINGS_PATH = REPORTS_DIR / "phase2_eda_findings.md"
EDA_SUMMARY_PATH = REPORTS_DIR / "phase2_eda_summary.json"

TARGET_COLUMN = "placement_status"
ID_COLUMNS = ["student_id"]
POST_OUTCOME_COLUMNS = ["salary_package_lpa"]
EXCLUDED_FAIRNESS_COLUMNS = ["age", "gender"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_MAPPING = {"Not Placed": 0, "Placed": 1}

EXPECTED_SHAPE = (100_000, 26)
EXPECTED_COLUMNS = [
    "student_id", "age", "gender", "cgpa", "branch", "college_tier",
    "internships_count", "projects_count", "certifications_count",
    "coding_skill_score", "aptitude_score", "communication_skill_score",
    "logical_reasoning_score", "hackathons_participated", "github_repos",
    "linkedin_connections", "mock_interview_score", "attendance_percentage",
    "backlogs", "extracurricular_score", "leadership_score",
    "volunteer_experience", "sleep_hours", "study_hours_per_day",
    "placement_status", "salary_package_lpa",
]

# The dataset has 23 candidate predictors after removing the identifier and post-outcome salary.
# Age and gender are additionally excluded from model features for fairness and relevance reasons.
MODEL_EXCLUDED_COLUMNS = ID_COLUMNS + POST_OUTCOME_COLUMNS + EXCLUDED_FAIRNESS_COLUMNS
MODEL_FEATURES = [c for c in EXPECTED_COLUMNS if c not in MODEL_EXCLUDED_COLUMNS + [TARGET_COLUMN]]
CATEGORICAL_FEATURES = ["branch", "college_tier", "volunteer_experience"]
NUMERICAL_FEATURES = [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]
EDA_CATEGORICAL_COLUMNS = ["gender", "branch", "college_tier", "volunteer_experience"]
EDA_NUMERICAL_COLUMNS = [
    c for c in EXPECTED_COLUMNS
    if c not in ID_COLUMNS + POST_OUTCOME_COLUMNS + [TARGET_COLUMN]
    and c not in EDA_CATEGORICAL_COLUMNS
]

for _directory in (PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
