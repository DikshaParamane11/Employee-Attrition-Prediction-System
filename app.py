import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# -------------------- LOAD MODEL --------------------
model = pickle.load(open('model (1).pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# -------------------- LOAD DATASET (FOR DASHBOARD) --------------------
df = pd.read_csv("HR_comma_sep.csv")

# -------------------- TITLE --------------------
st.title("📊 Employee Attrition Prediction System")
st.markdown("Predict whether an employee is likely to leave the company.")
st.markdown("---")

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Dashboard"])

# ==================== TAB 1: PREDICTION ====================
with tab1:

    st.info("Enter employee details from the sidebar to get prediction.")

    # -------- SIDEBAR INPUTS --------
    st.sidebar.header("📝 Enter Employee Details")

    satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.sidebar.slider("Last Evaluation", 0.0, 1.0, 0.5)
    number_project = st.sidebar.slider("Number of Projects", 1, 10, 3)
    average_montly_hours = st.sidebar.slider("Monthly Hours", 90, 300, 160)
    time_spend_company = st.sidebar.slider("Years in Company", 1, 10, 3)
    work_accident = st.sidebar.selectbox("Work Accident", [0, 1])
    promotion_last_5years = st.sidebar.selectbox("Promotion in Last 5 Years", [0, 1])

    salary = st.sidebar.selectbox("Salary", ["low", "medium", "high"])
    department = st.sidebar.selectbox("Department", [
        "sales", "technical", "support", "IT", "hr",
        "accounting", "marketing", "product_mng",
        "RandD", "management"
    ])

    # -------- INPUT PROCESSING --------
    input_data = [0] * len(columns)

    input_dict = {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'Work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years
    }

    for key in input_dict:
        if key in columns:
            input_data[columns.get_loc(key)] = input_dict[key]

    # One-hot encoding
    salary_col = f"salary_{salary}"
    dept_col = f"Department_{department}"

    if salary_col in columns:
        input_data[columns.get_loc(salary_col)] = 1

    if dept_col in columns:
        input_data[columns.get_loc(dept_col)] = 1

    # -------- LAYOUT --------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Employee Input Summary")
        st.write(f"Satisfaction Level: {satisfaction_level}")
        st.write(f"Projects: {number_project}")
        st.write(f"Monthly Hours: {average_montly_hours}")
        st.write(f"Years in Company: {time_spend_company}")
        st.write(f"Salary: {salary}")
        st.write(f"Department: {department}")

    with col2:
        st.subheader("📊 Prediction Result")

        if st.button("Predict"):
            prediction = model.predict([input_data])
            probability = model.predict_proba([input_data])[0][1]

            if prediction[0] == 1:
                st.error(f"⚠️ High Risk: Employee may leave\n\nProbability: {probability:.2f}")
            else:
                st.success(f"✅ Low Risk: Employee will stay\n\nProbability: {probability:.2f}")

            st.progress(int(probability * 100))

# ==================== TAB 2: DASHBOARD ====================
with tab2:

    st.subheader("📊 Employee Insights Dashboard")

    # -------- CHART 1 --------
    st.write("### Attrition Distribution")
    fig1, ax1 = plt.subplots()
    df['left'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title("Employee Attrition Count")
    st.pyplot(fig1)

    # -------- CHART 2 --------
    st.write("### Salary vs Attrition")
    fig2, ax2 = plt.subplots()
    pd.crosstab(df['salary'], df['left']).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    # -------- CHART 3 --------
    st.write("### Satisfaction Level vs Attrition")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='left', y='satisfaction_level', data=df, ax=ax3)
    st.pyplot(fig3)