# Fraud Detection Project: Comprehensive Summary & Study Guide

## 1. Project Overview
This project implements an end-to-end machine learning pipeline for credit card fraud detection. The goal is to accurately identify fraudulent transactions from a highly imbalanced dataset while minimizing false positives, and to aggregate these predictions to profile and flag high-risk sellers. The final output includes a tuned predictive model and the necessary artifacts to deploy a real-time Streamlit dashboard for monitoring.

## 2. Data Loading & Exploratory Data Analysis (EDA)
**What was done:** 
* Loaded the primary dataset (`transactions 2.csv`).
* Parsed date and time features (converting `transaction_time` to proper datetime objects).
* Analyzed summary statistics, revealing a strong class imbalance: only ~2.2% of transactions are fraudulent.
* Visualized distributions, such as transaction amounts (raw and log-transformed) and fraud rates across different channels and merchant categories.

**Why it was done:** 
Understanding the data distribution and identifying the severe class imbalance are critical first steps. It informs the choice of evaluation metrics (PR-AUC over ROC-AUC) and the need for specialized sampling or weighting techniques during modeling.

## 3. Feature Engineering
**What was done:**
* **Temporal Features:** Extracted `hour` and created a `night_flag` boolean to capture the intuition that fraud often occurs at odd hours.
* **Geospatial Features:** Created a `country_match` flag to check if the user's location matches the transaction location.
* **Transformations:** Applied logarithmic transformations (`log_amount`) to handle the heavy right-skew of transaction amounts.
* **Categorical Encoding:** Applied target encoding to high-cardinality categorical variables (saved as `target_encoding_maps.json`).

**Why it was done:**
Raw transactional data is rarely sufficient for complex ML models. Feature engineering explicitly exposes behavioral patterns (e.g., late-night transactions, location mismatches) to the model, drastically improving its predictive power.

## 4. Modeling Pipeline & Imbalance Handling
**What was done:**
* Built a baseline Logistic Regression model.
* Compared class imbalance strategies: Class Weighting vs. SMOTE. SMOTE showed a marginal improvement (+0.0042 PR-AUC) over class weighting on Logistic Regression.
* Evaluated advanced tree-based models: Random Forest (RF) and XGBoost.

**Why it was done:**
Establishing a baseline helps quantify the value of complex models. Because the dataset is highly imbalanced, standard accuracy is misleading; hence, models were optimized and evaluated primarily using **PR-AUC** (Precision-Recall Area Under Curve) and **F1-Score**.

## 5. Final Model Selection & Leaderboard
**What was done:**
* Compiled a final leaderboard of all models evaluated on the validation set.
* **Winner:** The tuned **XGBoost (XGB Tuned)** model significantly outperformed the others.
* **Metrics for Winner:** PR-AUC = 0.8572, ROC-AUC = 0.9708, F1 (Fraud) = 0.4500.

**Why it was done:**
A rigorous validation leaderboard ensures that the chosen model actually generalizes well to unseen data, preventing overfitting. XGBoost's ability to handle non-linear relationships and interactions made it the superior choice.

## 6. Model Interpretation (SHAP)
**What was done:**
* Calculated SHAP (SHapley Additive exPlanations) values to determine feature importance.
* Identified the top 3 drivers of fraud predictions: `shipping_distance_km`, `avs_match`, and `account_age_days`.

**Why it was done:**
In financial and fraud contexts, "black-box" models are unacceptable. SHAP provides global and local interpretability, allowing business stakeholders to understand *why* a transaction was flagged, building trust in the system.

## 7. Model Calibration & Threshold Tuning
**What was done:**
* **Calibration:** Evaluated model confidence using Expected Calibration Error (ECE). The uncalibrated XGB model had an ECE of 0.0931. Platt Scaling and Isotonic Regression were tested, with Isotonic achieving perfect calibration (ECE=0.0). Platt scaling was appended to the model pipeline.
* **Threshold Tuning:** Shifted the classification threshold from the default 0.50 to an optimal **0.90**. This improved the F1 score from 0.4500 to 0.7918 while balancing Precision (0.7827) and Recall (0.8011).

**Why it was done:**
* **Calibration:** Ensures that a predicted probability of 80% actually means there is an 80% chance of fraud, which is vital for downstream risk scoring.
* **Threshold Tuning:** Fraud detection is highly sensitive to the business cost of False Positives (upsetting real customers) versus False Negatives (losing money). Tuning the threshold optimizes this specific trade-off.

## 8. User-Level Risk Profiling (Seller Layer)
**What was done:**
* Aggregated individual transaction predictions to the user (seller) level.
* Created behavioral risk metrics: `avg_risk_score`, `high_risk_fraction`, and `risk_trend_slope` (detecting escalating risk over time).
* Established logic to flag "high-risk" sellers if they breached specific thresholds (e.g., `avg_risk_score >= 0.4`).

**Why it was done:**
Fraud is often systematic. While identifying a single fraudulent transaction is good, identifying a compromised or malicious *seller account* allows the platform to freeze the account, preventing future losses at scale.

## 9. Productionization & Dashboard Prep
**What was done:**
* Exported all required model artifacts for production: `model.pkl`, `scaler.pkl`, `feature_names.json`, `config.json`, `target_encoding_maps.json`, and `avg_amount_map.json`.
* Sampled a `demo_transactions.csv` (2000 rows, 25% fraud rate artificially inflated for demonstration) to simulate live data.
* Prepared the system to run via a Streamlit UI (`streamlit run dashboard.py`).

**What it changes:**
This bridges the gap between a static Jupyter Notebook experiment and a deployable software product. By decoupling the trained model and feature maps into artifacts, the system is now ready to be integrated into a live, real-time transaction streaming environment (like Kafka) and visualized via Streamlit.
