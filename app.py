
import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# LOAD MODEL
with open("model.pkl",'rb') as f:
    model = pickle.load(f)

# TITLE
st.title("👨‍💼 Employee Attrition Prediction")
st.markdown(
    "Predict whether an employee is likely to **Stay** or **Leave** "
    "the company using the trained Machine Learning model."
)

st.divider()



# SIDEBAR
st.sidebar.header("Employee Information")

st.sidebar.markdown(
    """
    Enter the employee details below.

    The model will predict:
    - 🟢 Stayed
    - 🔴 Left
    """
)



# INPUT FEATURES
# -------------------- NUMERICAL FEATURES --------------------

st.subheader("📊 Employee Details")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=59,
        value=30,
        step=1
    )

    years_at_company = st.number_input(
        "Years at Company",
        min_value=1,
        max_value=51,
        value=5,
        step=1
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1316.0,
        max_value=16149.0,
        value=5000.0,
        step=500.0
    )


with col2:

    number_promotions = st.number_input(
        "Number of Promotions",
        min_value=0,
        max_value=4,
        value=1,
        step=1
    )

    distance_from_home = st.number_input(
        "Distance from Home",
        min_value=1.0,
        max_value=99.0,
        value=10.0,
        step=1.0
    )

    number_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=6,
        value=2,
        step=1
    )


with col3:

    company_tenure = st.number_input(
        "Company Tenure",
        min_value=2,
        max_value=128,
        value=5,
        step=1
    )

# CATEGORICAL FEATURES

st.subheader("👤 Personal & Job Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    job_role = st.selectbox(
        "Job Role",
    [
        "Technology",
        "Healthcare",
        "Education",
        "Media",
        "Finance"
    ]
)

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


with col2:

    remote_work = st.selectbox(
        "Remote Work",
        [
            "Yes",
            "No"
        ]
    )

    overtime = st.selectbox(
        "Overtime",
        [
            "Yes",
            "No"
        ]
    )

    leadership_opportunities = st.selectbox(
        "Leadership Opportunities",
        [
            "Yes",
            "No"
        ]
    )


with col3:

    innovation_opportunities = st.selectbox(
        "Innovation Opportunities",
        [
            "Yes",
            "No"
        ]
    )


# ORDINAL FEATURES
st.subheader("⭐ Employee Ratings")

col1, col2, col3, col4 = st.columns(4)


with col1:

    work_life_balance = st.selectbox(
        "Work-Life Balance",
        [
            "Poor",
            "Fair",
            "Good",
            "Excellent"
        ]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )


with col2:

    performance_rating = st.selectbox(
        "Performance Rating",
        [
            "Below Average",
            "Low",
            "Average",
            "High"
        ]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "High School",
            "Associate Degree",
            "Bachelor's Degree",
            "Master's Degree",
            "PhD"
        ]
    )


with col3:

    job_level = st.selectbox(
        "Job Level",
        [
            "Entry",
            "Mid",
            "Senior"
        ]
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )


with col4:

    company_reputation = st.selectbox(
        "Company Reputation",
        [
            "Poor",
            "Fair",
            "Good",
            "Excellent"
        ]
    )

    employee_recognition = st.selectbox(
        "Employee Recognition",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

# CREATE INPUT DATAFRAME
input_data = pd.DataFrame({

    "Age": [age],

    "Years at Company": [years_at_company],

    "Monthly Income": [monthly_income],

    "Number of Promotions": [number_promotions],

    "Distance from Home": [distance_from_home],

    "Number of Dependents": [number_dependents],

    "Company Tenure": [company_tenure],

    "Gender": [gender],

    "Job Role": [job_role],

    "Marital Status": [marital_status],

    "Remote Work": [remote_work],

    "Overtime": [overtime],

    "Leadership Opportunities": [leadership_opportunities],

    "Innovation Opportunities": [innovation_opportunities],

    "Work-Life Balance": [work_life_balance],

    "Job Satisfaction": [job_satisfaction],

    "Performance Rating": [performance_rating],

    "Education Level": [education_level],

    "Job Level": [job_level],

    "Company Size": [company_size],

    "Company Reputation": [company_reputation],

    "Employee Recognition": [employee_recognition]
})


# PREDICTION BUTTON
st.divider()

predict_button = st.button(
    "🔮 Predict Employee Attrition",
    type="primary",
    use_container_width=True
)

# PREDICTION
if predict_button:

    try:

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability
        probability = model.predict_proba(input_data)[0]

        leave_probability = probability[1]
        stay_probability = probability[0]


        # RESULT
        st.subheader("📌 Prediction Result")


        if prediction == 1:

            st.error(
                "🔴 **Employee is likely to Leave**"
            )

            st.metric(
                "Probability of Leaving",
                f"{leave_probability * 100:.2f}%"
            )


        else:

            st.success(
                "🟢 **Employee is likely to Stay**"
            )

            st.metric(
                "Probability of Staying",
                f"{stay_probability * 100:.2f}%"
            )

        # PROBABILITY BREAKDOWN
        st.subheader("📊 Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Stayed",
                f"{stay_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Left",
                f"{leave_probability * 100:.2f}%"
            )


        # Progress bar for attrition probability

        st.write("Probability of Employee Leaving")

        st.progress(
            float(leave_probability)
        )

        # INTERPRETATION
        st.subheader("💡 Interpretation")

        if leave_probability >= 0.70:

            st.warning(
                "The model indicates a relatively high predicted "
                "probability of employee attrition."
            )

        elif leave_probability >= 0.40:

            st.info(
                "The model indicates a moderate predicted "
                "probability of employee attrition."
            )

        else:

            st.success(
                "The model indicates a relatively low predicted "
                "probability of employee attrition."
            )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


