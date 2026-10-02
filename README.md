# 🚆 RailGuard AI

### Railway Safety Anomaly Detection, Risk Intelligence & Predictive Analytics

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://navpreet-kaurr-railguard-ai-app-8h6u9e.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-black?logo=github)](https://github.com/Navpreet-kaurr/RailGuard-AI)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)

> **RailGuard AI** is a historical railway safety intelligence system that combines data analysis, machine learning, anomaly detection, risk classification, baseline forecasting, and railway infrastructure funding analysis into an interactive dashboard.

---

## 🌐 Live Demo

### 🚀 [Open RailGuard AI Dashboard](https://navpreet-kaurr-railguard-ai-app-8h6u9e.streamlit.app/)

Explore the deployed dashboard to interact with:

- Railway accident trend analysis
- Anomaly detection
- Historical risk intelligence
- Baseline forecasting
- Railway station funding analysis
- Interactive visualizations

---

## 📌 Overview

RailGuard AI analyzes historical railway safety data published by the **Government of India Open Government Data (OGD) Platform**.

The system focuses on identifying historical patterns and unusual changes in consequential train accidents while providing transparent analytical indicators for historical risk and a simple baseline estimate for the next period.

It also analyzes railway-zone-wise station development and maintenance funding to provide additional infrastructure-related insights.

### Key objectives

- Analyze historical consequential train accident trends
- Detect statistically unusual accident patterns
- Classify completed years using a transparent historical risk indicator
- Generate a simple next-period accident baseline
- Analyze railway station funding allocation and expenditure
- Build an interactive data analytics dashboard
- Use official government datasets instead of fabricated data

---

# 🧠 Analytical & Machine Learning Components

## 1. 🔍 Isolation Forest — Anomaly Detection

RailGuard AI uses **Isolation Forest** to identify historically unusual accident patterns.

The model considers:

- Consequential train accidents
- Year-to-year accident change
- Percentage change from the previous year

The analysis identifies statistically unusual observations around:

| Financial Year | Observation |
|---|---|
| **2020-21** | Statistical anomaly |
| **2021-22** | Statistical anomaly |

These observations represent statistical deviations from the historical pattern and **should not be interpreted as causal explanations**.

---

## 2. ⚠️ Historical Risk Classification

Because the annual accident dataset is small, RailGuard AI uses a **transparent rule-based historical indicator** instead of training a classification model on a very limited dataset.

Each completed year is classified by comparing its accident count with the historical average.

Risk categories:

- 🔴 **High**
- 🟠 **Medium**
- 🟢 **Low**

> **Important:** This is a historical analytical indicator and **not a probability of future railway accidents**.

---

## 3. 🔮 Baseline Forecasting

RailGuard AI uses a simple historical baseline rather than a complex forecasting model because the available annual dataset is small.

The next-period estimate is calculated using the average of the **three most recent completed financial years**.

```text
2021-22 → 35
2022-23 → 48
2023-24 → 40
────────────────
Baseline → 41.0
