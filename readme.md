# ⚡ End-to-End Energy Data Forecasting Pipeline

An automated, end-to-end machine learning pipeline for **day-ahead electricity price forecasting** using ENTSO-E market data. The project focuses on a **production-ready architecture** that automates data ingestion, feature engineering, model training, and next-day electricity price prediction.

---

## 📌 Project Overview

The pipeline continuously updates itself by combining historical electricity market data with newly available ENTSO-E data and forecast information.

It automatically:

- Extracts and cleans historical electricity market data.
- Fetches the latest actual market data from the ENTSO-E API.
- Downloads day-ahead forecasts for load and renewable generation.
- Dynamically engineers model features.
- Retrains the forecasting model using the latest available data.
- Predicts electricity prices for the next trading day.

---

# 🏗️ Pipeline Workflow

```text
Historical Data
       │
       ▼
Data Extraction & Cleaning
       │
       ▼
Master Dataset Creation
       │
       ▼
Feature Engineering
       │
       ▼
Model Training
       │
       ▼
Download Latest Actual Data
       │
       ▼
Update Historical Dataset
       │
       ▼
Download Next-Day Forecast Inputs
(Load + Solar + Wind)
       │
       ▼
Generate Forecast Features
       │
       ▼
Predict Day-Ahead Electricity Prices
```

---

## 🚀 Features

- 📥 Automated historical data ingestion
- 🌍 Live ENTSO-E API integration
- ⚡ Day-ahead electricity price forecasting
- 📈 Dynamic feature engineering
- 🔄 Automatic model retraining
- 🌤 Renewable generation forecast integration
- 🔋 Load forecast integration
- 💾 Lakehouse-style Parquet storage
- 🤖 LightGBM-based machine learning model

---

```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

> **Note:** On some systems, you may need to use:

```bash
python3 -m venv .venv
```

or

```bash
py -m venv .venv
```

---

## 3. Activate the Virtual Environment

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

(Optional) Update scikit-learn if required:

```bash
pip install --upgrade scikit-learn
```

---

# ▶️ Running the Pipeline

Execute the complete forecasting workflow using:

```bash
python app.py
```

The pipeline will automatically:

1. Load historical datasets
2. Download the latest ENTSO-E actual data
3. Update the historical database
4. Engineer model features
5. Retrain the forecasting model
6. Download next-day forecast inputs
7. Generate day-ahead electricity price forecasts

---

# 📊 Feature Engineering

The model automatically generates features including:

- Calendar features
  - Hour
  - Day of Week
  - Month
  - Day of Year
  - Weekend indicator

- Cyclical encodings
  - Hour (sin/cos)
  - Month (sin/cos)

- Load-based features

- Renewable generation features
  - Solar
  - Wind Onshore
  - Wind Offshore

- Residual load

- Historical lag features
  - 24-hour
  - 48-hour
  - 168-hour

- Rolling statistics
  - 24-hour rolling mean
  - 24-hour rolling standard deviation

---

# 📈 Machine Learning Model

The forecasting model is built using **LightGBM**, trained on dynamically engineered historical features and continuously updated with newly available market data.

---

# 📦 Data Sources

The pipeline utilizes data from the **ENTSO-E Transparency Platform**, including:

- Day-Ahead Electricity Prices
- Total Load
- Load Forecast
- Wind Forecast
- Solar Forecast
- Generation Data

---

# 💾 Data Storage

Processed datasets are stored in a partitioned Lakehouse structure:

```text
data/
└── lakehouse/
    ├── day_ahead_prices/
    │   └── zone=DE_LU/
    │       └── year=YYYY/
    │           └── month=MM/
    │
    ├── total_load/
    │   └── zone=DE_LU/
    │
    └── generation/
        └── zone=DE_LU/
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- LightGBM
- Scikit-learn
- ENTSO-E API
- Matplotlib
- Parquet

---

# 📄 License

This project is intended for educational and self learning purposes.