# 🚆 RailGuard AI

## Railway Safety Anomaly Detection, Risk Intelligence & Predictive Analytics

RailGuard AI is a data analytics and machine learning project designed to analyze historical railway safety data from the **Government of India Open Government Data (OGD) Platform**.

The system analyzes consequential train accidents, identifies unusual historical patterns, provides a historical risk classification, generates a simple baseline forecast, and analyzes railway station funding allocation and expenditure across railway zones.

> **Note:** This project is designed for historical safety intelligence and exploratory analysis. It is not a real-time accident prediction or operational railway safety system.

---

## 🎯 Project Objectives

- Analyze historical consequential train accident trends.
- Detect unusual changes in accident patterns.
- Classify historical accident years into risk levels.
- Generate a simple next-period accident baseline.
- Analyze railway station development and maintenance funding.
- Visualize insights through an interactive Streamlit dashboard.
- Use official Government of India datasets rather than fabricated data.

---

## 📊 Government Data Sources

### 1. Consequential Train Accidents

**Dataset:** Year-wise Number of Consequential Train Accidents, 2014-15 to 2024-25

**Source:** Government of India Open Government Data (OGD) Platform, Ministry of Railways.

**Dataset:**  
https://www.data.gov.in/resource/year-wise-number-consequential-train-accidents-2014-15-2024-25

The dataset contains year-wise consequential train accident counts from 2014-15 to 2024-25.

> **Important:** The 2024-25 observation is reported only up to October 2024 and is therefore treated as a partial-year observation.

---

### 2. Railway Station Funding

**Dataset:** Zonal-wise Details of Funds Allocated and Expenditure by Indian Railway for Development and Maintenance of Stations from 2020-21 to 2022-23

**Source:** Government of India Open Government Data (OGD) Platform.

**Dataset:**  
https://www.data.gov.in/resource/zonal-wise-details-funds-allocated-and-expenditure-indian-railway-development-and

The dataset contains railway-zone-wise allocation and expenditure information for railway station development and maintenance.

---

## 🧠 Machine Learning & Analytical Methods

### 1. Isolation Forest — Anomaly Detection

Isolation Forest is used to identify historically unusual accident patterns.

The model uses:

- Consequential train accidents
- Year-to-year accident change
- Percentage change from the previous year

The model identifies observations that differ substantially from the historical pattern.

The analysis identifies unusual changes around:

- **2020-21**
- **2021-22**

These are statistical anomalies and should not be interpreted as causal explanations.

---

### 2. Historical Risk Classification

A transparent rule-based historical risk indicator is used instead of training a classification model on the extremely small annual dataset.

Risk levels are calculated by comparing each completed year's accident count with the historical average.

Risk categories:

- **High**
- **Medium**
- **Low**

This is a historical indicator and **not a probability of future accidents**.

---

### 3. Baseline Forecasting

The project uses a simple historical baseline to estimate the next-period accident count.

The forecast is calculated using the average of the three most recent completed financial years.

The calculation is:

```text
2021-22 → 35
2022-23 → 48
2023-24 → 40

Baseline estimate → 41.0