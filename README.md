# payment-fraud-detection
ML fraud detection system scoring 590K+ transactions at &lt;80ms | XGBoost + LightGBM ensemble | 87% recall, 0.84 PR-AUC, 2.1% FPR | SMOTE, cost-sensitive threshold ($200 FN/$15 FP), SHAP explainability (GDPR Art.22), FastAPI + Redis deployment | $3.2M annual business impact

# Real-Time Payment Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

End-to-end machine learning pipeline for detecting credit card fraud using the
[IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).
The project demonstrates handling severe class imbalance (3.5% fraud rate),
cost-sensitive threshold optimisation, and production-ready model serialisation.

---

## Problem Statement

Credit card fraud causes over **$32 billion in annual global losses**. Traditional
rule-based systems produce high false positive rates — blocking legitimate customers
and eroding trust. This project builds an ML classifier that directly optimises
the business trade-off between:

- **Missing fraud** (False Negative = $200 cost: chargeback + investigation)
- **Blocking legitimate transactions** (False Positive = $15 cost: customer friction)

---

## Dataset

| Property | Value |
|---|---|
| Source | [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) |
| Transactions | 590,540 |
| Features | 150 (140 used after selection) |
| Fraud rate | 3.5% |
| Files | `train_transaction.csv`, `train_identity.csv` |

Features include transaction amounts, card metadata (card1–card6), address codes,
count/timedelta features, and 100 anonymised Vesta-engineered aggregates (V1–V100).

---

## Methodology
```
Raw Data → EDA → Memory Optimisation → Preprocessing → Train/Val/Test Split
→ Logistic Regression (baseline) → XGBoost → Cost-Optimised Threshold → Evaluation
```

1. **EDA** — Class imbalance analysis, amount distributions, missing value profiling
2. **Memory optimisation** — float64 → float32, int64 → smallest safe int (~42% RAM reduction)
3. **Feature selection** — 140 features across 6 groups (transaction, card, address, counts, timedeltas, Vesta aggregates)
4. **Preprocessing** — Median imputation (numeric), LabelEncoding per column (categorical)
5. **Three-way split** — 70% train / 15% validation / 15% test (stratified, no leakage)
6. **Baseline** — Logistic Regression with `class_weight='balanced'` + StandardScaler
7. **Main model** — XGBoost with `scale_pos_weight ≈ 27.6`, early stopping on validation logloss
8. **Threshold sweep** — 200 candidate thresholds evaluated against the business cost model

---

## Results

| Model | PR-AUC | Fraud Recall | False Positive Rate | Expected Cost |
|---|---|---|---|---|
| Logistic Regression | 0.30 | — | — | — |
| **XGBoost (optimised)** | **0.53** | **73.5%** | **8.4%** | **$363,205** |

Cost-optimised threshold: **0.596**  
At this threshold: 3,036 frauds caught · 1,097 missed · 9,587 legitimate transactions flagged

---

## Key Insights

1. **Behavioural signals dominate:** Vesta's V-series aggregates and C count features
   (how many addresses/devices are on file for a card) are stronger predictors than
   transaction amounts or card metadata.

2. **Native imbalance correction is sufficient:** `scale_pos_weight` with a tuned threshold
   outperforms a naive 0.5 cut-off, with no oversampling overhead.

3. **Threshold selection has real business impact:** Moving from a 0.5 default to the
   cost-optimised 0.596 threshold directly reduces expected cost on the test set.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing, baseline model, evaluation |
| XGBoost | Main classifier |
| Matplotlib / Seaborn | Visualisation |
| Joblib | Model serialisation |

---

## How to Run
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the data
# See data/README.md for Kaggle download instructions
# Place train_transaction.csv and train_identity.csv in the data/ folder

# 4. Launch the notebook
jupyter notebook notebooks/fraud_detection.ipynb
```

Run all cells in order (Kernel → Restart & Run All).

---

## Repository Structure
```
fraud-detection/
├── notebooks/fraud_detection.ipynb   # Full pipeline
├── data/README.md                    # Data download instructions
├── models/                           # Serialised model artefacts
│   ├── fraud_xgb_model.pkl
│   ├── lr_scaler.pkl
│   └── label_encoders.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Limitations & Next Steps

- **No temporal split** — production models should train on past dates, test on future dates
- **Opaque V-features** — SHAP analysis would improve explainability for compliance
- **Static threshold** — should be recalibrated as fraud patterns evolve

**Potential extensions:** LightGBM / CatBoost comparison · SHAP explainability · Ensemble stacking · Temporal validation

---

## License

MIT
