import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #312e81 0%, #4f46e5 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}


/* ---------- Main Header ---------- */

.dashboard-header {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 35px;
    border-radius: 22px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.20);
}

.dashboard-header h1 {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 8px;
}

.dashboard-header p {
    font-size: 17px;
    margin: 0;
}


/* ---------- Metric Cards ---------- */

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
    min-height: 145px;
}

.metric-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #4f46e5;
}

.metric-label {
    font-size: 14px;
    color: #6b7280;
    margin-top: 6px;
}


/* ---------- Section Titles ---------- */

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #312e81;
    margin-top: 10px;
    margin-bottom: 18px;
}


/* ---------- Information Cards ---------- */

.info-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border-left: 5px solid #6366f1;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    min-height: 300px;
}

.info-card h3 {
    color: #3730a3;
    margin-bottom: 15px;
}

.info-card p {
    color: #4b5563;
    line-height: 1.7;
}


/* ---------- Technology Boxes ---------- */

.tech-box {
    background: #eef2ff;
    color: #3730a3;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 10px;
    font-weight: 600;
}


/* ---------- Workflow ---------- */

.workflow-card {
    background: white;
    padding: 15px 8px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    min-height: 135px;
}

.workflow-number {
    background: #4f46e5;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}

.workflow-icon {
    font-size: 25px;
    margin-top: 8px;
}

.workflow-title {
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    margin-top: 8px;
}


/* ---------- Buttons ---------- */

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9);
}


/* ---------- Prediction Result ---------- */

.prediction-card {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #c7d2fe;
    text-align: center;
}

.prediction-score {
    font-size: 42px;
    font-weight: 700;
    color: #4f46e5;
}


/* ---------- Mobile Friendly ---------- */

@media (max-width: 768px) {
    .dashboard-header h1 {
        font-size: 28px;
    }

    .dashboard-header p {
        font-size: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    with open("student_performance_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("student_performance.csv")


model_data = load_model()
pipeline = model_data["pipeline"]
metrics = model_data["metrics"]

df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Student Predictor")
st.sidebar.caption("Machine Learning Academic Support System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Project Modules",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🤖 Prediction",
        "📈 Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "B.Sc. Data Science Project • 2026–2027"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    # ---------- Header ----------

    st.title("🎓 Student Performance Prediction")

    st.write(
        "A data-driven machine learning system for predicting "
        "student academic performance."
    )

    st.markdown("---")

    # ---------- Metric Cards ----------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👩‍🎓 Students",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "📊 Prediction Features",
            "13"
        )

    with col3:
        st.metric(
            "🤖 ML Model",
            "Random Forest"
        )

    with col4:
        st.metric(
            "🎯 R² Score",
            metrics["R2"]
        )

    st.markdown("---")

    # ---------- About Project ----------

    st.subheader("📌 About the Project")

    st.write(
        """
        This project uses machine learning to predict student academic
        performance based on important academic, behavioural and
        demographic factors.
        """
    )

    st.write(
        """
        The system analyzes student information such as attendance,
        study hours, previous marks, assignment performance,
        participation and previous examination scores.
        """
    )

    st.write(
        """
        The goal is to provide an early indication of expected
        performance and help identify students who may require
        additional academic support.
        """
    )

    st.markdown("---")

    # ---------- Technology Stack ----------

    st.subheader("⚙️ Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.info("🐍 **Python**")
        st.info("🐼 **Pandas & NumPy**")

    with tech2:
        st.info("🤖 **Scikit-learn**")
        st.info("📊 **Matplotlib**")

    with tech3:
        st.info("🌐 **Streamlit**")
        st.info("📁 **CSV Dataset**")

    st.markdown("---")

    # ---------- Model Performance ----------

    st.subheader("📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Mean Absolute Error",
            metrics["MAE"]
        )

    with c2:
        st.metric(
            "Root Mean Squared Error",
            metrics["RMSE"]
        )

    with c3:
        st.metric(
            "R² Score",
            metrics["R2"]
        )

    st.markdown("---")

    # ---------- Project Workflow ----------

    st.subheader("🔄 Project Workflow")

    workflow = [
        "📥 Data Collection",
        "🧹 Data Preprocessing",
        "📊 Exploratory Data Analysis",
        "🎯 Feature Selection",
        "🤖 Model Training",
        "🔮 Performance Prediction",
        "📈 Model Evaluation"
    ]

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.info("**1. 📥 Data Collection**")
        st.info("**2. 🧹 Data Preprocessing**")

    with w2:
        st.info("**3. 📊 Exploratory Data Analysis**")
        st.info("**4. 🎯 Feature Selection**")

    with w3:
        st.info("**5. 🤖 Model Training**")
        st.info("**6. 🔮 Performance Prediction**")

    with w4:
        st.info("**7. 📈 Model Evaluation**")

    st.markdown("---")

    # ---------- Dataset Summary ----------

    st.subheader("📋 Dataset Summary")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.success("👩‍🎓 **500 Student Records**")

    with d2:
        st.success("📊 **15 Dataset Columns**")

    with d3:
        st.success("🎯 **Final Performance as Target**")

    st.markdown("---")

    st.caption(
        "⚠️ Predictions are estimates generated by the machine learning "
        "model and should not be considered a definitive assessment "
        "of a student."
    )

# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    st.caption(
        "Explore the student dataset and identify academic patterns."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Students",
        f"{df.shape[0]:,}"
    )

    c2.metric(
        "Total Columns",
        df.shape[1]
    )

    c3.metric(
        "Average Performance",
        f"{df['Final_Performance'].mean():.1f}%"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        [
            "📋 Dataset",
            "📈 Relationships",
            "📊 Statistics"
        ]
    )

    # ---------- Dataset ----------

    with tab1:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    # ---------- Relationships ----------

    with tab2:

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.scatter(
            df["Attendance"],
            df["Final_Performance"],
            alpha=0.65
        )

        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Final Performance (%)")
        ax.set_title("Attendance vs Final Performance")
        ax.grid(alpha=0.2)

        st.pyplot(fig)

        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.scatter(
            df["Study_Hours"],
            df["Final_Performance"],
            alpha=0.65
        )

        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Final Performance (%)")
        ax.set_title("Study Hours vs Final Performance")
        ax.grid(alpha=0.2)

        st.pyplot(fig)

        plt.close(fig)

    # ---------- Statistics ----------

    with tab3:

        st.dataframe(
            df.describe().round(2),
            use_container_width=True
        )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🤖 Prediction":

    st.title("🤖 Performance Prediction")

    st.caption(
        "Enter a student's details and generate an estimated final performance."
    )

    with st.form("prediction_form"):

        st.subheader("👩‍🎓 Student Information")

        c1, c2, c3 = st.columns(3)

        # ---------- Column 1 ----------

        with c1:

            age = st.number_input(
                "Age",
                min_value=17,
                max_value=30,
                value=20
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            department = st.selectbox(
                "Department",
                [
                    "Data Science",
                    "Computer Science",
                    "Information Technology",
                    "Commerce"
                ]
            )

            attendance = st.slider(
                "Attendance (%)",
                40.0,
                100.0,
                80.0
            )

        # ---------- Column 2 ----------

        with c2:

            study_hours = st.slider(
                "Study Hours per Day",
                1.0,
                10.0,
                4.5,
                0.5
            )

            previous_marks = st.slider(
                "Previous Marks (%)",
                35.0,
                100.0,
                70.0
            )

            assignment_score = st.slider(
                "Assignment Score (%)",
                35.0,
                100.0,
                75.0
            )

            participation = st.slider(
                "Participation (%)",
                20.0,
                100.0,
                70.0
            )

        # ---------- Column 3 ----------

        with c3:

            previous_exam = st.slider(
                "Previous Exam Score (%)",
                35.0,
                100.0,
                70.0
            )

            sleep_hours = st.slider(
                "Sleep Hours",
                4.0,
                9.0,
                7.0,
                0.5
            )

            internet = st.selectbox(
                "Internet Access",
                ["Yes", "No"]
            )

            extracurricular = st.selectbox(
                "Extracurricular Activity",
                ["Yes", "No"]
            )

            parental_support = st.selectbox(
                "Parental Support",
                ["High", "Medium", "Low"]
            )

        submitted = st.form_submit_button(
            "🔮 Predict Student Performance",
            use_container_width=True
        )

    # ---------- Prediction Result ----------

    if submitted:

        input_df = pd.DataFrame(
            [{
                "Age": age,
                "Gender": gender,
                "Department": department,
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Previous_Marks": previous_marks,
                "Assignment_Score": assignment_score,
                "Participation": participation,
                "Previous_Exam_Score": previous_exam,
                "Sleep_Hours": sleep_hours,
                "Internet_Access": internet,
                "Extracurricular_Activity": extracurricular,
                "Parental_Support": parental_support
            }]
        )

        prediction = float(
            np.clip(
                pipeline.predict(input_df)[0],
                0,
                100
            )
        )

        if prediction >= 85:

            category = "Excellent"
            message = (
                "The predicted performance is in the excellent range."
            )

        elif prediction >= 70:

            category = "Good"
            message = (
                "The predicted performance is in the good range."
            )

        elif prediction >= 50:

            category = "Average"
            message = (
                "The prediction indicates scope for further improvement."
            )

        else:

            category = "Needs Improvement"
            message = (
                "The prediction indicates that additional academic "
                "support may be useful."
            )

        st.markdown("---")

        a, b = st.columns(2)

        with a:

            st.markdown(
                f"""
                <div class="prediction-card">

                    <h3>🎯 Predicted Performance</h3>

                    <div class="prediction-score">
                        {prediction:.1f}%
                    </div>

                    <h4>{category}</h4>

                    <p>{message}</p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with b:

            st.subheader("📊 Prediction Result")

            st.metric(
                "Predicted Score",
                f"{prediction:.1f}%"
            )

            st.progress(
                int(round(prediction))
            )

            st.info(
                f"**Performance Category:** {category}"
            )

        st.caption(
            "This is a machine-learning estimate for academic analysis "
            "and should not be treated as a definitive assessment "
            "of a student's ability."
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "📈 Model Evaluation":

    st.title("📈 Model Evaluation")

    st.caption(
        "Performance of the trained Random Forest Regression model."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        metrics["MAE"]
    )

    c2.metric(
        "RMSE",
        metrics["RMSE"]
    )

    c3.metric(
        "R² Score",
        metrics["R2"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🤖 Model Details")

    st.write(
        "**Algorithm:** Random Forest Regression"
    )

    st.write(
        "**Training/Test Split:** 80% / 20%"
    )

    st.write(
        f"**Training Records:** {int(len(df) * 0.8):,}"
    )

    st.write(
        f"**Testing Records:** {len(df) - int(len(df) * 0.8):,}"
    )

    st.write(
        "**Preprocessing:** Numerical scaling + categorical one-hot encoding"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📚 Metric Interpretation")

    st.write(
        "**MAE:** Measures the average absolute prediction error. "
        "Lower values indicate smaller errors."
    )

    st.write(
        "**RMSE:** Gives greater weight to larger prediction errors. "
        "Lower values are generally better."
    )

    st.write(
        "**R² Score:** Indicates how much variation in final performance "
        "is explained by the model; values closer to 1 indicate a stronger fit."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎓 Student Performance Prediction Using Machine Learning • "
    "B.Sc. Data Science Academic Project • 2026–2027"
)