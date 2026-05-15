# Seller-Level Fraud Detection

**DATA 245 — Machine Learning Technologies, Spring 2026**  
Team: Tejas Sawant, Ameya Khond, Prathmesh Mankar

End-to-end ML pipeline for credit card fraud detection with a novel seller-level risk aggregation layer. Identifies fraudulent transactions and flags high-risk seller accounts for platform-level intervention.

---

## Dataset

`transactions 2.csv` — 299,695 transactions, 17 features, ~2.2% fraud rate, ~6,000 unique users/sellers.

---

## Pipeline Overview

| Stage | Key Work |
|---|---|
| EDA | Class imbalance analysis, distribution visualization |
| Feature Engineering | Temporal (`hour`, `night_flag`), geospatial (`country_match`), `log_amount`, target encoding |
| Imbalance Handling | Class weighting vs. SMOTE (SMOTE +0.0042 PR-AUC) |
| Modeling | Logistic Regression → Random Forest → XGBoost |
| Tuning | Stratified CV, hyperparameter search |
| Interpretation | SHAP values — top features: `shipping_distance_km`, `avs_match`, `account_age_days` |
| Calibration | Isotonic Regression (ECE 0.0931 → 0.0), threshold tuned to **0.90** |
| Seller Layer | `avg_risk_score`, `high_risk_fraction`, `risk_trend_slope` per seller |
| Dashboard | Streamlit real-time monitoring UI |

---

## Results

**Best model: XGBoost (tuned + threshold 0.90)**

| Metric | Score |
|---|---|
| PR-AUC | 0.8572 |
| ROC-AUC | 0.9708 |
| F1 (Fraud) | 0.7918 |
| Precision | 0.7827 |
| Recall | 0.8011 |

---

## Project Structure

```
fraud-detection.ipynb         # Full ML pipeline notebook
dashboard.py                  # Streamlit dashboard
summary.md                    # Detailed study guide / writeup

# Model artifacts (exported for production)
model.pkl
scaler.pkl
feature_names.json
config.json
target_encoding_maps.json
avg_amount_map.json

# Data
transactions 2.csv            # Full dataset
demo_transactions.csv         # 2,000-row demo (25% fraud, for dashboard)
```

---

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Novel Contribution

Beyond transaction-level predictions, the pipeline aggregates model outputs to the **seller level**, computing rolling average risk, high-risk transaction fraction, and risk trend slope. Sellers breaching thresholds (e.g., `avg_risk_score >= 0.4`) are flagged for account review — catching systematic fraud at scale rather than one transaction at a time.
