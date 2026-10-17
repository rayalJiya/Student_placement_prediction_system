"""Streamlit UI for the Student Placement Prediction System.

This app is an educational demo. Its underlying model has weak validation performance
and dataset provenance is unverified, so it must not be used for consequential decisions.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "selected_placement_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Placement Prediction System")
st.caption("Educational demonstration — not a guarantee of a student's placement outcome.")

st.warning(
    "**Important limitation:** the current model has weak test discrimination (ROC-AUC about 0.585), "
    "and the dataset's source and label-generation process have not been verified. "
    "Do not use this result to deny opportunities, rank students, or make high-stakes decisions."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model. Check that Phase 3 artifacts exist. Details: {exc}")
    st.stop()

expected_features = metadata.get("model_features", [])
if not expected_features:
    st.error("Model metadata does not list model_features. Cannot safely build the input form.")
    st.stop()

st.markdown("### Enter a student's pre-placement details")
st.write("Complete every field. No student details are pre-filled; the app only predicts after you provide the inputs.")

with st.form("student_profile_form"):
    st.markdown("#### Academics and institution")
    c1, c2, c3 = st.columns(3)
    with c1:
        cgpa = st.number_input("CGPA (4.5–10.0)", min_value=4.5, max_value=10.0, value=None, step=0.1, placeholder="Enter CGPA")
        branch = st.selectbox("Branch", ["— Select —", "CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"])
    with c2:
        college_tier = st.selectbox("College tier", ["— Select —", "Tier 1", "Tier 2", "Tier 3"])
        backlogs = st.number_input("Current / historical backlogs (0–6)", min_value=0, max_value=6, value=None, step=1, placeholder="Enter count")
    with c3:
        attendance_percentage = st.number_input("Attendance percentage (50–100)", min_value=50.0, max_value=100.0, value=None, step=1.0, placeholder="Enter percentage")
        study_hours_per_day = st.number_input("Study hours per day (0.5–10)", min_value=0.5, max_value=10.0, value=None, step=0.5, placeholder="Enter hours")

    st.markdown("#### Experience and portfolio")
    c1, c2, c3 = st.columns(3)
    with c1:
        internships_count = st.number_input("Internships (0–8)", min_value=0, max_value=8, value=None, step=1, placeholder="Enter count")
        projects_count = st.number_input("Projects (0–13)", min_value=0, max_value=13, value=None, step=1, placeholder="Enter count")
        certifications_count = st.number_input("Certifications (0–11)", min_value=0, max_value=11, value=None, step=1, placeholder="Enter count")
    with c2:
        hackathons_participated = st.number_input("Hackathons (0–8)", min_value=0, max_value=8, value=None, step=1, placeholder="Enter count")
        github_repos = st.number_input("GitHub repositories (0–16)", min_value=0, max_value=16, value=None, step=1, placeholder="Enter count")
        linkedin_connections = st.number_input("LinkedIn connections (50–999)", min_value=50, max_value=999, value=None, step=10, placeholder="Enter count")
    with c3:
        volunteer_experience = st.selectbox("Volunteer experience", ["— Select —", "Yes", "No"])
        extracurricular_score = st.number_input("Extracurricular score (0–100)", min_value=0.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")
        leadership_score = st.number_input("Leadership score (0–100)", min_value=0.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")

    st.markdown("#### Skills and interview preparation (scores 20–100)")
    c1, c2, c3 = st.columns(3)
    with c1:
        coding_skill_score = st.number_input("Coding skill score", min_value=20.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")
        aptitude_score = st.number_input("Aptitude score", min_value=20.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")
    with c2:
        communication_skill_score = st.number_input("Communication skill score", min_value=20.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")
        logical_reasoning_score = st.number_input("Logical reasoning score", min_value=20.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")
    with c3:
        mock_interview_score = st.number_input("Mock interview score", min_value=20.0, max_value=100.0, value=None, step=1.0, placeholder="Enter score")
        sleep_hours = st.number_input("Sleep hours per night (3–10)", min_value=3.0, max_value=10.0, value=None, step=0.5, placeholder="Enter hours")

    submitted = st.form_submit_button("Predict placement status", type="primary", use_container_width=True)

if submitted:
    raw_values = {
        "cgpa": cgpa,
        "branch": branch,
        "college_tier": college_tier,
        "internships_count": internships_count,
        "projects_count": projects_count,
        "certifications_count": certifications_count,
        "coding_skill_score": coding_skill_score,
        "aptitude_score": aptitude_score,
        "communication_skill_score": communication_skill_score,
        "logical_reasoning_score": logical_reasoning_score,
        "hackathons_participated": hackathons_participated,
        "github_repos": github_repos,
        "linkedin_connections": linkedin_connections,
        "mock_interview_score": mock_interview_score,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "extracurricular_score": extracurricular_score,
        "leadership_score": leadership_score,
        "volunteer_experience": volunteer_experience,
        "sleep_hours": sleep_hours,
        "study_hours_per_day": study_hours_per_day,
    }
    blank = [k for k, v in raw_values.items() if v is None or (isinstance(v, str) and v == "— Select —")]
    if blank:
        st.error("Please complete all fields before requesting a prediction.")
        st.write("Missing fields:", ", ".join(blank))
    else:
        # Construct input in the exact feature order stored in metadata; never include target,
        # student ID, salary, age, or gender.
        row = pd.DataFrame([{name: raw_values[name] for name in expected_features}], columns=expected_features)
        try:
            prediction = model.predict(row)[0]
            result = str(prediction)
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            placed_index = classes.index("Placed")
            placed_score = float(probabilities[placed_index])
            st.markdown("### Model output")
            if result == "Placed":
                st.success("Model prediction: **Placed**")
            else:
                st.info("Model prediction: **Not Placed**")
            st.metric("Model-estimated score for Placed", f"{placed_score:.1%}")
            st.caption("This is the model's estimated probability, not a calibrated guarantee of real-world placement.")
            st.progress(max(0.0, min(1.0, placed_score)))
            if placed_score >= 0.70:
                st.write("Illustrative score band: **Higher model score**")
            elif placed_score >= 0.40:
                st.write("Illustrative score band: **Middle model score**")
            else:
                st.write("Illustrative score band: **Lower model score**")
            st.markdown("#### General preparation ideas")
            ideas = []
            if float(coding_skill_score) < 70: ideas.append("Practice coding fundamentals and timed problem-solving.")
            if float(aptitude_score) < 70: ideas.append("Practice quantitative aptitude and logical reasoning regularly.")
            if float(communication_skill_score) < 70: ideas.append("Practice explaining projects clearly and conducting mock interviews.")
            if float(mock_interview_score) < 70: ideas.append("Review mock interview feedback and rehearse role-specific questions.")
            if int(internships_count) == 0: ideas.append("Look for relevant internships, virtual experience, or practical client-style projects.")
            if int(projects_count) < 3: ideas.append("Build and document a few complete projects with clear problem statements and measurable evaluation.")
            if ideas:
                for idea in ideas: st.write(f"- {idea}")
            else:
                st.write("Keep practicing and document your work; these inputs alone cannot guarantee a placement outcome.")
            st.warning("This is an educational prototype. The model's validation performance is weak and the dataset provenance is unverified. Do not use this score for selection, exclusion, or high-stakes decisions.")
        except Exception as exc:
            st.error(f"Prediction failed. Check model and feature configuration. Details: {exc}")

with st.expander("Model and fairness notes"):
    st.write(f"Model: {metadata.get('selected_model', 'Unknown')}")
    st.write("Inputs match the saved model's feature list. Student ID and post-placement salary are excluded to avoid identifier use and target leakage.")
    st.write("Age and gender are excluded from model inputs in this project. That does not prove the model is fair; indirect proxy effects may remain.")
    st.write("The underlying test ROC-AUC was approximately 0.585, so the model has limited ability to separate placed from not-placed students.")
