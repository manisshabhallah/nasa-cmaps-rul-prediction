# PHM Framework — NASA C-MAPSS
## M.Tech Data Engineering | Manisha Bhalla (M25DE1050)

**End-to-End Prognostics & Health Management for Turbofan Engines**

---

## What This Project Does

This project builds a complete predictive maintenance system for aircraft engines.
Given 21 sensor readings from a turbofan engine over time, it predicts how many
operational cycles remain before the engine needs maintenance — this is called
Remaining Useful Life (RUL).

The system follows a Medallion architecture (Bronze → Silver → Gold), trains a
deep LSTM neural network, explains predictions with SHAP, and shows everything
in a live Streamlit dashboard.

---

## Project Structure

```
phm_project/
│
├── data/
│   ├── raw/          ← NASA .txt files go here (you download these)
│   ├── bronze/       ← Parquet files after raw ingestion
│   ├── silver/       ← Cleaned, normalized, labeled Parquet
│   └── gold/         ← Sliding window .npy arrays for training
│
├── models/           ← Saved LSTM weights + scalers
├── outputs/          ← Plots, metrics CSVs, SHAP charts
│
├── src/
│   ├── bronze_layer.py   ← Step 1: Raw data → Parquet
│   ├── silver_layer.py   ← Step 2: Clean + RUL labels + normalize
│   ├── gold_layer.py     ← Step 3: Sliding windows + lag features
│   ├── model.py          ← LSTM architecture definition
│   ├── train.py          ← Training loop with early stopping
│   ├── evaluate.py       ← RMSE, MAE, NASA score, plots
│   ├── explain.py        ← SHAP feature importance
│   └── inference.py      ← Real-time RUL prediction
│
├── dashboard/
│   └── app.py            ← Streamlit dashboard (5 pages)
│
├── run_pipeline.py       ← ONE script to run everything
├── download_data.py      ← Downloads NASA C-MAPSS dataset
└── requirements.txt
```

---

## How to Run (Step by Step)

### Step 0 — Set up environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 1 — Download the dataset

```bash
python download_data.py
```

If the auto-download fails, manually download from:
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Download and place these files in `data/raw/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

### Step 2 — Run the full pipeline

```bash
# Run everything (takes ~3-5 minutes on CPU)
python run_pipeline.py --dataset FD001

# To skip SHAP (faster, save ~2 min):
python run_pipeline.py --dataset FD001 --skip_shap

# Custom training settings:
python run_pipeline.py --dataset FD001 --epochs 100 --batch_size 128 --patience 15
```

### Step 3 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open browser at: http://localhost:8501

---

## Run Individual Steps (Optional)

If you want to run one step at a time:

```bash
# 1. Bronze layer (raw ingestion)
python src/bronze_layer.py

# 2. Silver layer (cleaning + labeling)
python src/silver_layer.py

# 3. Gold layer (feature engineering)
python src/gold_layer.py

# 4. Train the model
python src/train.py --dataset FD001 --epochs 50

# 5. Evaluate
python src/evaluate.py --dataset FD001

# 6. Fleet inference
python src/inference.py --dataset FD001

# 7. SHAP explanations
python src/explain.py --dataset FD001
```

---

## What Each File Does (Simple Explanation)

| File | What it does |
|------|-------------|
| `bronze_layer.py` | Reads raw .txt files → saves as Parquet. Adds metadata like timestamp. Like receiving a package — you log it exactly as it arrived. |
| `silver_layer.py` | Cleans the data, removes useless sensors, normalizes values to 0-1, computes RUL labels using piecewise method. Like washing and sorting the package contents. |
| `gold_layer.py` | Creates 30-cycle sliding windows and adds delta/rolling features. This is what the LSTM actually trains on. |
| `model.py` | Defines the LSTM neural network architecture. Two stacked LSTM layers + two Dense layers. |
| `train.py` | Trains the model using Adam optimizer + early stopping. Saves the best checkpoint. Plots loss curves. |
| `evaluate.py` | Loads saved model, runs on test set, computes RMSE/MAE/NASA score, saves plots. |
| `explain.py` | Uses SHAP to rank which sensors matter most for predictions. |
| `inference.py` | Given any new sensor sequence → outputs predicted RUL + Health Index + alert level. |
| `dashboard/app.py` | Streamlit web app showing all outputs interactively. |

---

## The Data — NASA C-MAPSS

NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset:

- FD001: 100 training engines, 1 fault mode, 1 operating condition
- FD002: 260 training engines, 1 fault mode, 6 operating conditions

Each row = one operational cycle of one engine.
21 sensor columns (temperature, pressure, speed, etc.)
Goal = predict how many cycles until maintenance needed.

---

## Expected Results

After training on FD001 for 50 epochs:

| Metric | Expected |
|--------|----------|
| RMSE   | ~13–18 cycles |
| MAE    | ~10–15 cycles |
| MAPE   | ~20–35% |

Target from NASA 2015 paper: RMSE ≈ 12.5 (with more advanced methods).

---

## Dashboard Pages

1. **Fleet Overview** — All engines ranked by health. Red/yellow/green status. Scatter plot.
2. **Engine Deep Dive** — Pick one engine, see sensor trends + RUL trajectory over time.
3. **Model Performance** — Loss curves, scatter plot, per-engine bar chart, metrics table.
4. **SHAP Explainability** — Which sensors caused the prediction? Bar chart + table.
5. **Live Prediction** — Upload your own CSV or pick a test engine → get RUL instantly.

---

## References

1. Saxena et al. (2015). Review and Analysis of Algorithmic Approaches for C-MAPSS.
   NASA/TM-2015-218764

2. Predictive Maintenance of Turbofan Engine using NASA-CMAPSS Dataset (2024)
   https://www.researchgate.net/publication/397406031

3. Enhancing Aircraft Safety with LSTM Health Monitoring (2024)
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11154484
