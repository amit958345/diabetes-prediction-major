# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# PROFESSIONAL STREAMLIT WEB APPLICATION
# ==========================================


# ------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------

import os
import sys

import pandas as pd
import joblib

import streamlit as st


# ------------------------------------------
# 2. FIND PROJECT ROOT DIRECTORY
# ------------------------------------------

# app.py is inside the "app" folder.
# Moving one level upward gives us the
# main diabetes-ml-research project folder.

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ------------------------------------------
# 3. ADD SRC DIRECTORY TO PYTHON PATH
# ------------------------------------------

# The saved model uses the custom IQRClipper
# transformer located inside src/iqr_clipper.py.

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.append(SRC_PATH)


# Import the custom transformer so that
# joblib can correctly load the saved model.

from iqr_clipper import IQRClipper


# ------------------------------------------
# 4. LOAD SAVED MACHINE LEARNING MODEL
# ------------------------------------------

# Create the complete path to the saved model.

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "diabetes_model.pkl"
)


# Load the trained preprocessing + ML pipeline.

model = joblib.load(
    MODEL_PATH
)


# ------------------------------------------
# 5. STREAMLIT PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(

    page_title="Diabetes Prediction",

    page_icon="🩺",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ------------------------------------------
# 6. CUSTOM CSS
# ------------------------------------------

# CSS improves the appearance of the
# Streamlit application.

st.markdown(
    """
    <style>

    /* Main application area */

    .main {
        padding-top: 1rem;
    }


    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }


    /* Subtitle */

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }


    /* Section headings */

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* Prediction result cards */

    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #dddddd;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }


    /* Prediction heading */

    .prediction-heading {
        font-size: 26px;
        font-weight: 700;
    }


    /* Probability value */

    .probability-value {
        font-size: 32px;
        font-weight: 700;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #777777;
        padding: 30px 0 10px 0;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------
# 7. SIDEBAR
# ------------------------------------------

with st.sidebar:

    # Sidebar title

    st.title("🩺 Diabetes ML")


    # Project information

    st.markdown(
        """
        ### About the Project

        **Diabetes Prediction Using
        Machine Learning**

        This academic project evaluates
        multiple machine learning
        classification algorithms using
        the Pima Indians Diabetes Dataset.
        """
    )


    # Horizontal divider

    st.divider()


    # Selected model information

    st.markdown(
        """
        **Selected Model**

        Logistic Regression
        """
    )


    # Input feature information

    st.markdown(
        """
        **Input Features**

        8 health-related features
        """
    )


    # Another divider

    st.divider()


    # Academic project label

    st.caption(
        "Academic & Research Project"
    )


# ------------------------------------------
# 8. MAIN HEADER
# ------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🩺 Diabetes Prediction Using '
    'Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# Subtitle below the main heading

st.markdown(
    '<div class="subtitle">'
    'A machine-learning-based academic '
    'system for diabetes risk prediction'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------
# 9. PROJECT INTRODUCTION
# ------------------------------------------

st.markdown(
    '<div class="section-title">'
    '📋 About the System'
    '</div>',
    unsafe_allow_html=True
)


# Short explanation of the application

st.info(
    """
    Enter the required health-related
    information below. The trained machine
    learning pipeline will process the input
    and generate a prediction.
    """
)


# ------------------------------------------
# 10. PATIENT INFORMATION
# ------------------------------------------

st.markdown(
    '<div class="section-title">'
    '👤 Patient Information'
    '</div>',
    unsafe_allow_html=True
)


# Create two columns so that the eight
# input features are displayed neatly.

left_column, right_column = st.columns(
    2,
    gap="large"
)


# ------------------------------------------
# 11. LEFT COLUMN INPUTS
# ------------------------------------------

with left_column:

    # Number of pregnancies

    pregnancies = st.number_input(

        "Pregnancies",

        min_value=0,

        max_value=20,

        value=2,

        step=1,

        help="Number of times pregnant."
    )


    # Glucose level

    glucose = st.number_input(

        "Glucose",

        min_value=0.0,

        max_value=250.0,

        value=120.0,

        step=1.0,

        help="Plasma glucose concentration."
    )


    # Blood pressure

    blood_pressure = st.number_input(

        "Blood Pressure",

        min_value=0.0,

        max_value=150.0,

        value=70.0,

        step=1.0,

        help="Diastolic blood pressure."
    )


    # Skin thickness

    skin_thickness = st.number_input(

        "Skin Thickness",

        min_value=0.0,

        max_value=100.0,

        value=25.0,

        step=1.0,

        help="Triceps skin fold thickness."
    )


# ------------------------------------------
# 12. RIGHT COLUMN INPUTS
# ------------------------------------------

with right_column:

    # Insulin level

    insulin = st.number_input(

        "Insulin",

        min_value=0.0,

        max_value=900.0,

        value=80.0,

        step=1.0,

        help="Serum insulin level."
    )


    # Body Mass Index

    bmi = st.number_input(

        "BMI",

        min_value=0.0,

        max_value=70.0,

        value=30.5,

        step=0.1,

        help="Body Mass Index."
    )


    # Diabetes pedigree function

    diabetes_pedigree = st.number_input(

        "Diabetes Pedigree Function",

        min_value=0.0,

        max_value=3.0,

        value=0.45,

        step=0.01,

        help="Diabetes pedigree function value."
    )


    # Age

    age = st.number_input(

        "Age",

        min_value=1,

        max_value=120,

        value=35,

        step=1,

        help="Age in years."
    )


# ------------------------------------------
# 13. PREDICTION BUTTON
# ------------------------------------------

# Add some spacing before the button.

st.write("")


# Prediction button

predict_button = st.button(

    "🔍 Predict Diabetes Risk",

    type="primary",

    use_container_width=True
)


# ------------------------------------------
# 14. RUN PREDICTION
# ------------------------------------------

# This block executes only when the user
# clicks the prediction button.

if predict_button:


    # --------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------

    # Create a DataFrame containing the
    # eight features required by the model.

    # IMPORTANT:
    # Column names must exactly match the
    # columns used during model training.

    input_data = pd.DataFrame({

        "Pregnancies": [pregnancies],

        "Glucose": [glucose],

        "BloodPressure": [blood_pressure],

        "SkinThickness": [skin_thickness],

        "Insulin": [insulin],

        "BMI": [bmi],

        "DiabetesPedigreeFunction":
            [diabetes_pedigree],

        "Age": [age]
    })


    # --------------------------------------
    # MODEL PREDICTION
    # --------------------------------------

    # Predict the class:
    #
    # 0 = lower predicted risk
    # 1 = predicted diabetes risk

    prediction = model.predict(
        input_data
    )


    # --------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------

    # Get probabilities for both classes.

    probability = model.predict_proba(
        input_data
    )


    # Probability of Class 0

    probability_no_diabetes = (
        probability[0][0]
    )


    # Probability of Class 1

    probability_diabetes = (
        probability[0][1]
    )


    # --------------------------------------
    # RESULT SECTION
    # --------------------------------------

    st.divider()


    # Prediction result heading

    st.markdown(
        '<div class="section-title">'
        '📊 Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------
    # PREDICTION MESSAGE
    # --------------------------------------

    # Check the predicted class.

    if prediction[0] == 1:

        # Class 1 prediction

        st.error(
            "⚠️ The model predicts a higher "
            "diabetes risk."
        )

    else:

        # Class 0 prediction

        st.success(
            "✅ The model predicts a lower "
            "diabetes risk."
        )


    # --------------------------------------
    # PROBABILITY CARDS
    # --------------------------------------

    # Create two columns for the
    # probability cards.

    result_left, result_right = st.columns(
        2,
        gap="large"
    )


    # --------------------------------------
    # CLASS 0 CARD
    # --------------------------------------

    with result_left:

        st.markdown(
            '<div class="prediction-card">'
            '<div class="prediction-heading">'
            'Class 0'
            '</div>'
            '<p>Lower Predicted Risk</p>'
            f'<div class="probability-value">'
            f'{probability_no_diabetes:.2%}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------
    # CLASS 1 CARD
    # --------------------------------------

    with result_right:

        st.markdown(
            '<div class="prediction-card">'
            '<div class="prediction-heading">'
            'Class 1'
            '</div>'
            '<p>Predicted Diabetes Risk</p>'
            f'<div class="probability-value">'
            f'{probability_diabetes:.2%}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------
    # PROBABILITY VISUALIZATION
    # --------------------------------------

    st.subheader(
        "📈 Prediction Probability"
    )


    # --------------------------------------
    # CLASS 0 PROBABILITY
    # --------------------------------------

    st.markdown(
        f"""
        **Class 0 — Lower Predicted Risk**

        **{probability_no_diabetes:.2%}**
        """
    )


    # Progress bar for Class 0 probability.

    st.progress(
        probability_no_diabetes
    )


    # --------------------------------------
    # CLASS 1 PROBABILITY
    # --------------------------------------

    st.markdown(
        f"""
        **Class 1 — Predicted Diabetes Risk**

        **{probability_diabetes:.2%}**
        """
    )


    # Progress bar for Class 1 probability.

    st.progress(
        probability_diabetes
    )


    # --------------------------------------
    # INPUT SUMMARY
    # --------------------------------------

    st.subheader(
        "📋 Input Summary"
    )


    # Display the values entered by the user.

    st.dataframe(

        input_data,

        use_container_width=True,

        hide_index=True
    )


# ------------------------------------------
# 15. PROJECT METHODOLOGY
# ------------------------------------------

# Divider before methodology.

st.divider()


# Methodology heading

st.markdown(
    '<div class="section-title">'
    '🔬 Project Methodology'
    '</div>',
    unsafe_allow_html=True
)


# Display the complete workflow.

st.markdown(
    """
    **Dataset → Data Quality Analysis → EDA
    → Preprocessing → Model Training
    → Model Comparison → Cross-Validation
    → Final Model → Prediction**
    """
)


# ------------------------------------------
# 16. MODEL INFORMATION
# ------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🤖 Model Information'
    '</div>',
    unsafe_allow_html=True
)


# Create three columns for model metrics.

model_col1, model_col2, model_col3 = (
    st.columns(3)
)


# ------------------------------------------
# SELECTED MODEL
# ------------------------------------------

with model_col1:

    st.metric(
        "Selected Model",
        "Logistic Regression"
    )


# ------------------------------------------
# CROSS-VALIDATION F1
# ------------------------------------------

with model_col2:

    st.metric(
        "CV F1-Score",
        "64.50%"
    )


# ------------------------------------------
# CROSS-VALIDATION ROC-AUC
# ------------------------------------------

with model_col3:

    st.metric(
        "CV ROC-AUC",
        "83.95%"
    )


# ------------------------------------------
# 17. ACADEMIC DISCLAIMER
# ------------------------------------------

st.divider()


st.warning(
    """
    ⚠️ **Academic and Research Disclaimer**

    This system is developed for academic
    and research purposes. The prediction
    generated by the machine learning model
    is not a medical diagnosis and should
    not replace professional medical advice.
    """
)


# ------------------------------------------
# 18. FOOTER
# ------------------------------------------

st.markdown(
    '<div class="footer">'
    'Diabetes Prediction Using Machine Learning'
    ' • Academic Project • NIST University'
    '</div>',
    unsafe_allow_html=True
)