# 📊 Employee Attrition Prediction System

## 📌 Project Overview

This project is a **Machine Learning-based system** that predicts whether an employee is likely to **leave or stay** in a company.
It helps HR teams identify high-risk employees and take preventive actions.

---

## 🎯 Objective

* Predict employee attrition using historical data
* Identify key factors affecting employee turnover
* Provide insights for better HR decision-making

---

## 📂 Dataset

The project uses the **HR Analytics dataset**, which includes features like:

* Satisfaction level
* Last evaluation
* Number of projects
* Monthly working hours
* Time spent in company
* Salary and department
* Work accident & promotion

**Target Variable:**

* `0` → Employee stays
* `1` → Employee leaves

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Model:** Random Forest
* **Deployment:** Streamlit

---

## 🔍 Project Workflow

1. Data Loading & Understanding
2. Data Cleaning (duplicate removal)
3. Exploratory Data Analysis (EDA)
4. Feature Engineering & Encoding
5. Train-Test Split (Stratified)
6. Model Training (Random Forest)
7. Hyperparameter Tuning
8. Cross Validation (Stratified K-Fold)
9. Model Evaluation (Accuracy, Recall, etc.)
10. Deployment using Streamlit

---

## 🤖 Model Details

* Algorithm: **Random Forest Classifier**
* Accuracy: ~98%
* Used **class_weight='balanced'** to handle class imbalance
* Applied **Stratified Cross-Validation** for stability

---

## 📊 Key Insights

* Low satisfaction → Higher attrition
* Low salary → Higher attrition
* Workload imbalance → Attrition risk
* Department impacts employee behavior

---

## 🚀 Deployment

The model is deployed using **Streamlit**:

* User enters employee details
* Model predicts **Stay / Leave**
* Displays prediction with **probability score**
* Includes basic dashboard visualizations

---

## ▶️ How to Run the Project

```bash
# Clone the repository
git clone <your-repo-link>

# Navigate to project folder
cd employee_attrition_project

# Activate virtual environment (if used)
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
employee_attrition_project/
│
├── app.py
├── employee_prediction_system.py
├── model.pkl
├── columns.pkl
├── HR_comma_sep.csv
├── requirements.txt
└── README.md
```

---

## 🧠 Key Learnings

* End-to-end ML pipeline development
* Handling class imbalance
* Model tuning and validation
* Deployment using Streamlit
* Converting ML models into real-world applications

---

## 📌 Future Improvements

* Add explainable AI (SHAP)
* Improve recall using threshold tuning
* Deploy using cloud platforms (AWS/Render)

---

## 👤 Author

**Diksha Paramane**

---

# ⭐ If you like this project, give it a star!
