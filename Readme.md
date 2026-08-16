# 🚀 NASA Turbofan Engine RUL Prediction (C-MAPSS)

An end-to-end Machine Learning and Predictive Maintenance pipeline designed to estimate the **Remaining Useful Life (RUL)** of NASA turbofan engine datasets (`FD001` through `FD004`). 

This project processes multi-stage sensor data, generates feature-engineered parquet stores, and presents actionable model insights via an interactive Streamlit dashboard.

---

## 📌 Features

* **Data Pipeline (Bronze Layer):** Preprocesses raw engine sensor metrics and structures them into optimized Parquet files (`RUL` and `test` suites for FD001–FD004).
* **Predictive Analytics:** Models degradation trends and estimates RUL using machine learning algorithms.
* **Interactive Dashboard:** Built with Streamlit (`app.py`) to visualize real-time engine health, sensor trends, and RUL estimations.

---

## 📁 Repository Structure

```text
MTP2_nasa/
├── dashboard/
│   ├── app.py          # Streamlit UI dashboard application
│   └── ux              # UX layout configuration and styling
├── data/
│   ├── bronze/         # Processed Parquet data stores (FD001–FD004)
│   └── ux              # Data UX schema definitions
└── ux                  # Main project configuration metadata
