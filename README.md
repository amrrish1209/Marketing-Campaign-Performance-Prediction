# Marketing Analytics & Business Intelligence

A comprehensive end-to-end data science and predictive modeling project designed to analyze, clean, and model e-commerce marketing campaign performance data across three major beauty platforms: **Nykaa**, **Purplle**, and **Tira**.

This repository covers initial ingestion, multi-platform data reconciliation, robust feature engineering, exploratory statistical testing, and both continuous (Regression) and categorical (Classification) machine learning pipelines.

---

## 📌 Project Overview
E-commerce marketing campaigns generate vast numbers of daily data points across scattered operational channels. The objective of this project is twofold:
1. **Predict Revenue (Regression):** Accurately forecast numerical campaign earnings to manage marketing budgets.
2. **Predict Profitability (Classification):** Classify campaigns dynamically into **Profit** or **Loss** categories to flag underperforming or high-risk marketing strategies before capital deployment.

---

## 🛠️ Tech Stack & Key Libraries
* **Language:** Python 3.x
* **Core Processing:** `pandas`, `numpy`, `scipy`
* **Machine Learning:** `scikit-learn`
* **Preprocessing Elements:** `Pipeline`, `ColumnTransformer`, `RobustScaler`, `StandardScaler`, `SimpleImputer`, `MultiLabelBinarizer`, `PolynomialFeatures`

---

## 📂 Project Workflow & Architecture

### 1. Data Ingestion & Cleansing
* Consolidates dirty campaign trackers from Nykaa, Purplle, and Tira containing varying missing value ratios.
* Employs standard string conditioning and splits multi-value comma-separated strings inside the categorical `channel_used` column via a vectorized `MultiLabelBinarizer` setup.

### 2. Analytical Feature Engineering
* **Data Leakage Defenses:** Target encoding matrices map distributions exclusively within cross-validation train splits to prevent statistical bleed-through.
* **Profitability Flagging:** Constructs the classification target `profit_loss_flag`. Rather than standard boundary targets ($ROI > 0$), a conservative operational margin threshold is applied ($ROI > 0.10$) to account for invisible corporate overhead costs and naturally alleviate severe class imbalances.

### 3. Exploratory Statistical Hypothesis Testing
Automated inference tests filter out non-informative columns prior to model construction:
* **Numerical Metrics:** Checked against target variables via Two-Sample $T$-tests and Analysis of Variance (ANOVA).
* **Categorical / Discrete Attributes:** Checked using Chi-Square ($\chi^2$) Contingency Tests.

### 4. Advanced Machine Learning Pipelines

#### 📈 Regression (Target: `revenue`)
The continuous pipeline evaluates four variations built over robust imputation and feature transformers:
* **Linear Regression:** Standard baseline approach.
* **Polynomial Regression (Degree 2):** Captures multi-variable interaction terms safely while preventing high-degree overfitting.
* **Decision Tree Regressor:** Non-linear modeling configured with a restricted depth threshold.
* **Random Forest Regressor:** An ensemble tree method running multiple parallelestimators to lower variance.

#### 🎯 Classification (Target: `profit_loss_flag`)
Categorical predictive structures built over standard scalers to bypass data skewing:
* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**

> **Note on Imbalance Resolution:** Classifiers are trained utilizing `class_weight='balanced'` parameters. This explicitly forces the cost function to treat minority class misses with high penalty values, preserving realistic Macro Average metrics and tracking vital target Recall metrics.

---

## 📊 Performance Metrology

### Regression Metrics
Models are tracked against training and testing segments using:
* **Mean Absolute Error (MAE)**
* **Coefficient of Determination ($R^2$ Score)**

### Classification Metrics
Evaluation emphasizes structural reliability instead of simple raw accuracy through:
* **Precision:** Minimizing false marketing alarms.
* **Recall:** Ensuring 100% of operational budget losses are successfully identified.
* **Macro F1-Score:** Tracking balanced class capabilities.

---

## 🚀 Getting Started

### Prerequisites
Install the required packages using pip:
```bash
pip install pandas numpy scipy scikit-learn
