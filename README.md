# 👨‍💼 Employee Attrition Prediction

A complete **Machine Learning project for Employee Attrition Prediction**, covering Exploratory Data Analysis (EDA), feature preprocessing, multiple classification algorithms, hyperparameter tuning, model evaluation, and a **Streamlit web application** for prediction.

---

## 📌 Project Overview

Employee attrition is an important business problem for organizations because employee turnover can affect productivity, recruitment costs, team stability, and business performance.

The objective of this project is to analyze employee-related factors and build a machine learning model that predicts whether an employee is likely to:

- 🟢 **Stay**
- 🔴 **Leave**

The final tuned **XGBoost** model is saved as a pickle pipeline and integrated into a Streamlit application.

---

## 🎯 Problem Statement

> **Predict whether an employee is likely to stay with or leave the company based on demographic, job-related, compensation, satisfaction, work environment, and company-related factors.**

### Target Variable

The original `Attrition` column contains:

```text
Stayed → 0
Left   → 1
```

---

# 📊 Dataset

The dataset contains:

- **59,598 records**
- **24 columns**
- **23 original features + 1 target**
- **8 numerical columns**
- **16 categorical columns**
- No missing values were observed in the dataset
- `Employee ID` is used only as an identifier and is removed before modeling

### Features

| Feature | Type |
|---|---|
| Employee ID | Identifier |
| Age | Numerical |
| Gender | Categorical |
| Years at Company | Numerical |
| Job Role | Categorical |
| Monthly Income | Numerical |
| Work-Life Balance | Ordinal |
| Job Satisfaction | Ordinal |
| Performance Rating | Ordinal |
| Number of Promotions | Numerical |
| Overtime | Categorical |
| Distance from Home | Numerical |
| Education Level | Ordinal |
| Marital Status | Categorical |
| Number of Dependents | Numerical |
| Job Level | Ordinal |
| Company Size | Ordinal |
| Company Tenure | Numerical |
| Remote Work | Categorical |
| Leadership Opportunities | Categorical |
| Innovation Opportunities | Categorical |
| Company Reputation | Ordinal |
| Employee Recognition | Ordinal |
| Attrition | Target |

---

# 🔎 Exploratory Data Analysis

The EDA notebook covers the following stages:

### 1. Data Understanding
- Dataset shape
- Data types
- Descriptive statistics
- Missing-value analysis
- Duplicate analysis
- Target distribution

### 2. Univariate Analysis
- Numerical feature distributions
- Categorical feature distributions
- Histograms
- Count plots
- Box plots
- Outlier analysis

### 3. Bivariate Analysis
The relationship between employee attributes and attrition was analyzed using different categorical and numerical visualizations.

Examples include:

- Gender vs Attrition
- Job Role vs Attrition
- Work-Life Balance vs Attrition
- Job Satisfaction vs Attrition
- Performance Rating vs Attrition
- Overtime vs Attrition
- Education Level vs Attrition
- Marital Status vs Attrition
- Job Level vs Attrition
- Company Size vs Attrition
- Remote Work vs Attrition
- Company Reputation vs Attrition
- Employee Recognition vs Attrition

### 4. Multivariate Analysis
Multiple variables were analyzed together to understand patterns involving employee characteristics and attrition.

---

# 📈 Key EDA Observations

The following are descriptive observations from this dataset:

- The dataset contains **59,598 employees**.
- **Work-Life Balance** shows a noticeable relationship with attrition. Employees with a Poor work-life balance have a higher observed attrition percentage than employees with an Excellent work-life balance.
- Employees working **Overtime** have a higher observed attrition percentage than employees who do not work overtime.
- **Single employees** have a higher observed attrition percentage than Married and Divorced employees.
- **Entry-level employees** have a higher observed attrition percentage than Mid and Senior employees.
- **Senior-level employees** have a considerably lower observed attrition percentage in this dataset.
- Employees with **Remote Work = Yes** have a lower observed attrition percentage than employees with Remote Work = No.
- Employees with **Poor Company Reputation** have a higher observed attrition percentage than employees with Good or Excellent company reputation.
- Employees with **Leadership Opportunities = Yes** show a slightly lower observed attrition percentage than employees without such opportunities.
- Employees with **Innovation Opportunities = Yes** also show a slightly lower observed attrition percentage.
- Job Role has relatively similar attrition percentages across the five job roles in this dataset.
- The observed relationships are descriptive and **do not establish causation**.

---

# 🧹 Data Preprocessing

The target variable was encoded as:

```python
df["Attrition"] = df["Attrition"].map({
    "Stayed": 0,
    "Left": 1
})
```

`Employee ID` was removed because it is an identifier rather than a predictive feature.

```python
X = df.drop(columns=["Attrition", "Employee ID"])
y = df["Attrition"]
```

## Train-Test Split

The dataset was split using:

```text
Training data : 80%
Testing data  : 20%
Random state  : 42
Stratified    : Yes
```

Resulting shapes:

```text
X_train : 47,678 rows
X_test  : 11,920 rows
```

---

# ⚙️ Feature Engineering & Preprocessing

Features were divided into three groups.

## Numerical Features

```text
Age
Years at Company
Monthly Income
Number of Promotions
Distance from Home
Number of Dependents
Company Tenure
```

Numerical features were standardized using:

```python
StandardScaler()
```

## Nominal Categorical Features

```text
Gender
Job Role
Marital Status
Remote Work
Overtime
Leadership Opportunities
Innovation Opportunities
```

These were encoded using:

```python
OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
    drop="first"
)
```

## Ordinal Features

```text
Work-Life Balance
Job Satisfaction
Performance Rating
Education Level
Job Level
Company Size
Company Reputation
Employee Recognition
```

These were encoded using `OrdinalEncoder` with an explicitly defined category order.

For example:

```text
Work-Life Balance:
Poor < Fair < Good < Excellent

Job Satisfaction:
Low < Medium < High < Very High

Job Level:
Entry < Mid < Senior

Company Size:
Small < Medium < Large
```

All preprocessing steps were combined using a Scikit-learn `ColumnTransformer` and `Pipeline`.

---

# 🤖 Machine Learning Models

The following classification algorithms were implemented and evaluated:

1. K-Nearest Neighbors (KNN)
2. Decision Tree
3. Gaussian Naive Bayes
4. Logistic Regression
5. Support Vector Classifier (SVC)
6. Random Forest
7. XGBoost

Both baseline and tuned models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix

---

# 🔧 Hyperparameter Tuning

Hyperparameter optimization was performed using:

- `GridSearchCV`
- `RandomizedSearchCV`
- Cross-validation

The primary tuning metric used was:

```text
F1 Score
```

For XGBoost, randomized search was performed with:

```text
n_iter = 20
cv = 3
scoring = "f1"
random_state = 42
```

---

# 📊 Tuned Model Performance

The final tuned models were evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 71.22% | 69.18% | 71.17% | 70.16% | 78.27% |
| Decision Tree | 73.24% | 69.74% | 77.24% | 73.30% | 81.18% |
| Naive Bayes | 73.37% | 71.59% | 72.95% | 72.26% | 81.79% |
| Logistic Regression | 74.00% | 72.87% | 72.21% | 72.54% | 83.09% |
| SVC | 74.59% | 73.77% | 72.25% | 73.00% | 83.91% |
| Random Forest | 74.95% | 74.24% | 72.46% | 73.34% | 84.26% |
| **XGBoost** | **75.46%** | **74.14%** | **74.31%** | **74.23%** | **84.85%** |

---

# 🏆 Final Model — XGBoost

The tuned XGBoost classifier was selected as the final model used in the deployment application.

### Tuned Parameters

```text
n_estimators       = 385
max_depth          = 6
learning_rate      = 0.0300439
subsample          = 0.9649841
colsample_bytree   = 0.8873062
min_child_weight   = 4
gamma              = 0.1478168
```

### Final Test Performance

```text
Accuracy  : 75.46%
Precision : 74.14%
Recall    : 74.31%
F1 Score  : 74.23%
ROC-AUC   : 84.85%
```

### Confusion Matrix

```text
                 Predicted
              Stayed   Left
Actual Stayed   4783   1469
       Left     1456   4212
```

---

# 💾 Model Deployment

The complete trained XGBoost pipeline was saved as:

```text
model.pkl
```

The saved pipeline contains the preprocessing steps and trained classifier, allowing new employee information to pass through the same preprocessing workflow before prediction.

The model is loaded in the Streamlit application using:

```python
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

# 🌐 Streamlit Web Application

The project includes a Streamlit application:

```text
app.py
```

The application allows users to enter employee information through an interactive interface.

### Input categories

#### Employee Details
- Age
- Years at Company
- Monthly Income
- Number of Promotions
- Distance from Home
- Number of Dependents
- Company Tenure

#### Personal & Job Information
- Gender
- Job Role
- Marital Status
- Remote Work
- Overtime
- Leadership Opportunities
- Innovation Opportunities

#### Employee Ratings
- Work-Life Balance
- Job Satisfaction
- Performance Rating
- Education Level
- Job Level
- Company Size
- Company Reputation
- Employee Recognition

After clicking:

```text
🔮 Predict Employee Attrition
```

the application displays:

- Predicted outcome
- Probability of Staying
- Probability of Leaving
- Probability breakdown
- Attrition probability progress bar
- Basic interpretation of the predicted probability

---

# 📂 Project Structure

```text
employee-attrition/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
│
├── employee_attrition_eda_file(1).ipynb
└── emply_attrition_model_impl(1).ipynb
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `model.pkl` | Trained XGBoost preprocessing + prediction pipeline |
| `requirements.txt` | Python dependencies |
| `employee_attrition_eda_file(1).ipynb` | Complete EDA notebook |
| `emply_attrition_model_impl(1).ipynb` | ML implementation, tuning and evaluation |
| `README.md` | Project documentation |

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- SciPy

### Deployment

- Streamlit

### Model Serialization

- Pickle

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate to the project folder:

```bash
cd employee-attrition
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

After installing the dependencies, run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 🧪 Run the Notebooks

### EDA

Open:

```text
employee_attrition_eda_file(1).ipynb
```

This notebook contains the exploratory analysis and visualizations.

### Machine Learning

Open:

```text
emply_attrition_model_impl(1).ipynb
```

This notebook contains:

1. Data loading
2. Target encoding
3. Feature selection
4. Train-test splitting
5. Preprocessing pipelines
6. Model training
7. Hyperparameter tuning
8. Model evaluation
9. Final XGBoost model
10. Model serialization

---

# 📌 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Preprocessing
   ↓
Multiple Classification Models
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Final XGBoost Pipeline
   ↓
model.pkl
   ↓
Streamlit Application
   ↓
Employee Attrition Prediction
```

---

# ⚠️ Disclaimer

This project is intended for educational and demonstration purposes.

The relationships identified during EDA are observations from the dataset and should not be interpreted as causal relationships.

Model predictions are probabilistic estimates and should not be used as the sole basis for employment-related decisions.

---

# 👩‍💻 Author

**Nalla Prathiba**

**B.Tech – Computer Science and Engineering**

Interested in **Data Science, Machine Learning, and AI**.

---

## ⭐ If you found this project useful

Feel free to explore the notebooks, experiment with different models and hyperparameters, and improve the Streamlit application.
