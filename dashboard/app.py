"""
dashboard/app.py
----------------
Streamlit PHM Dashboard

Pages:
  1. Fleet Health Overview  — all engines, color-coded by alert level
  2. Engine Deep Dive       — single engine sensor trends + RUL history
  3. Model Performance      — loss curve, scatter plot, metrics table
  4. SHAP Explanations      — feature importance
  5. Live Prediction        — paste/upload raw sensor data and get RUL instantly

Run with:
    cd phm_project
    streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="PHM Dashboard — NASA C-MAPSS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────── CSS Styling ────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0.2rem;
    }
    .sub-header {font-size: 0.95rem; color: #666; margin-bottom: 1.5rem;}
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #4361ee;
    }
    .critical-card {border-left-color: #e63946 !important;}
    .warning-card  {border-left-color: #f4a261 !important;}
    .healthy-card  {border-left-color: #2a9d8f !important;}
    .stAlert > div {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────── Helpers ────────────────────────────────
RUL_CAP = 125
DATASET_OPTIONS = ["FD001", "FD002", "FD003", "FD004"]


@st.cache_resource
def load_artifacts(dataset_id):
    """Load model + scaler + feature meta (cached so it loads once)."""
    try:
        from src.model import LSTMModel
        gold_dir = "data/gold"
        meta = joblib.load(os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))
        scaler = joblib.load(f"models/scaler_{dataset_id}.pkl")
        model = LSTMModel(input_size=len(meta["feature_cols"]),
                          hidden_size=64, num_layers=2, dropout=0.2)
        model.load_state_dict(torch.load(f"models/lstm_{dataset_id}_best.pth",
                                         map_location="cpu"))
        model.eval()
        return model, scaler, meta, True
    except Exception as e:
        return None, None, None, False


def alert_color(level):
    return {"CRITICAL": "#e63946", "WARNING": "#f4a261", "HEALTHY": "#2a9d8f"}.get(level, "#888")


def health_gauge(hi_value, engine_id):
    """Plotly gauge chart for a single engine's Health Index."""
    color = "#2a9d8f" if hi_value > 60 else ("#f4a261" if hi_value > 30 else "#e63946")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hi_value,
        title={"text": f"Engine {engine_id}<br>Health Index", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 30],   "color": "#fde8e8"},
                {"range": [30, 60],  "color": "#fff3e0"},
                {"range": [60, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75, "value": hi_value
            }
        },
        number={"suffix": "%", "font": {"size": 22}}
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
    return fig


# ──────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg",
             width=100)
    st.markdown("## PHM Dashboard")
    st.markdown("*Turbofan Engine Prognostics*")
    st.divider()

    dataset_id = st.selectbox("Dataset", DATASET_OPTIONS, index=0)
    page = st.radio("Navigation", [
        "🏠 Fleet Overview",
        "🔍 Engine Deep Dive",
        "📊 Model Performance",
        "🧠 SHAP Explainability",
        "⚡ Live Prediction"
    ])
    st.divider()
    st.markdown("**M.Tech Data Engineering**")
    st.markdown("NASA C-MAPSS | LSTM RUL")


model, scaler, meta, model_loaded = load_artifacts(dataset_id)

# ──────────────────────────── PAGE 1: Fleet Overview ─────────────────
if page == "🏠 Fleet Overview":
    st.markdown('<div class="main-header">✈️ Fleet Health Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Remaining Useful Life predictions for all turbofan engines</div>',
                unsafe_allow_html=True)

    fleet_path = f"outputs/fleet_health_{dataset_id}.csv"

    if not os.path.exists(fleet_path):
        st.warning("⚠️ Fleet predictions not found. Run `python src/inference.py` first.")
    else:
        fleet = pd.read_csv(fleet_path)

        # Summary KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Engines", len(fleet))
        with col2:
            crit = (fleet["alert_level"] == "CRITICAL").sum()
            st.metric("🔴 Critical", crit, delta=f"{crit/len(fleet)*100:.0f}% of fleet")
        with col3:
            warn = (fleet["alert_level"] == "WARNING").sum()
            st.metric("🟡 Warning", warn)
        with col4:
            ok = (fleet["alert_level"] == "HEALTHY").sum()
            st.metric("🟢 Healthy", ok)

        st.divider()

        # Fleet scatter — Actual vs Predicted RUL
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.subheader("Predicted vs Actual RUL")
            fig = px.scatter(
                fleet, x="true_rul", y="predicted_rul",
                color="alert_level",
                color_discrete_map={
                    "CRITICAL": "#e63946", "WARNING": "#f4a261", "HEALTHY": "#2a9d8f"
                },
                hover_data=["engine_id", "health_index", "error"],
                labels={"true_rul": "Actual RUL (cycles)",
                        "predicted_rul": "Predicted RUL (cycles)"},
                title=f"Fleet RUL Prediction — {dataset_id}"
            )
            # Perfect line
            lim = max(fleet["true_rul"].max(), fleet["predicted_rul"].max()) + 10
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                          line=dict(color="black", dash="dash", width=1.5))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Alert Distribution")
            alert_counts = fleet["alert_level"].value_counts().reset_index()
            alert_counts.columns = ["level", "count"]
            fig2 = px.pie(alert_counts, values="count", names="level",
                          color="level",
                          color_discrete_map={
                              "CRITICAL": "#e63946", "WARNING": "#f4a261", "HEALTHY": "#2a9d8f"
                          },
                          hole=0.45)
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)

        # Fleet table — sorted by health index (worst first)
        st.subheader("Engine Fleet — Sorted by Risk")
        display_fleet = fleet.copy()
        display_fleet["Status"] = display_fleet["alert_level"].apply(
            lambda x: "🔴 CRITICAL" if x == "CRITICAL" else
                      ("🟡 WARNING" if x == "WARNING" else "🟢 HEALTHY")
        )
        st.dataframe(
            display_fleet[["engine_id", "health_index", "predicted_rul", "true_rul",
                            "error", "Status"]].rename(columns={
                "engine_id": "Engine",
                "health_index": "Health Index (%)",
                "predicted_rul": "Predicted RUL",
                "true_rul": "Actual RUL",
                "error": "Error (cycles)"
            }),
            use_container_width=True,
            height=300
        )


# ──────────────────────────── PAGE 2: Engine Deep Dive ───────────────
elif page == "🔍 Engine Deep Dive":
    st.markdown('<div class="main-header">🔍 Engine Deep Dive</div>', unsafe_allow_html=True)

    silver_path = f"data/silver/train_{dataset_id}_silver.parquet"
    if not os.path.exists(silver_path):
        st.warning("Run the pipeline first: `python run_pipeline.py`")
    else:
        train_silver = pd.read_parquet(silver_path)
        engine_ids = sorted(train_silver["engine_id"].unique())
        selected_engine = st.selectbox("Select Engine", engine_ids, index=0)

        eng_data = train_silver[train_silver["engine_id"] == selected_engine].copy()
        eng_data = eng_data.sort_values("cycle")

        # Engine summary
        total_life = eng_data["cycle"].max()
        final_rul  = eng_data["rul"].min()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Operational Life", f"{total_life} cycles")
        col2.metric("Min RUL (at EOL)", f"{final_rul} cycles")
        col3.metric("Degradation Phase Start",
                    f"Cycle {total_life - min(total_life, RUL_CAP)}")

        st.divider()

        # Sensor selection
        sensor_cols = [c for c in eng_data.columns if c.startswith("s") and c[1:].isdigit()]
        selected_sensors = st.multiselect(
            "Select sensors to plot",
            sensor_cols,
            default=sensor_cols[:4] if len(sensor_cols) >= 4 else sensor_cols
        )

        if selected_sensors:
            # Sensor trends
            st.subheader(f"Sensor Readings — Engine {selected_engine}")
            fig = make_subplots(rows=len(selected_sensors), cols=1,
                                shared_xaxes=True,
                                subplot_titles=selected_sensors)
            for i, sensor in enumerate(selected_sensors, 1):
                fig.add_trace(
                    go.Scatter(x=eng_data["cycle"], y=eng_data[sensor],
                               name=sensor, line=dict(width=1.5)),
                    row=i, col=1
                )
            fig.update_layout(height=120 * len(selected_sensors) + 80,
                              showlegend=False, title_text="")
            fig.update_xaxes(title_text="Cycle", row=len(selected_sensors), col=1)
            st.plotly_chart(fig, use_container_width=True)

        # RUL trajectory
        st.subheader(f"RUL Trajectory — Engine {selected_engine}")
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(
            x=eng_data["cycle"], y=eng_data["rul"],
            fill="tozeroy", mode="lines",
            line=dict(color="#4361ee", width=2),
            name="RUL"
        ))
        fig_rul.add_hline(y=20, line_dash="dash", line_color="#e63946",
                           annotation_text="Critical threshold (20 cycles)")
        fig_rul.add_hline(y=50, line_dash="dot", line_color="#f4a261",
                           annotation_text="Warning threshold (50 cycles)")
        fig_rul.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Remaining Useful Life (cycles)",
            height=350
        )
        st.plotly_chart(fig_rul, use_container_width=True)


# ──────────────────────────── PAGE 3: Model Performance ──────────────
elif page == "📊 Model Performance":
    st.markdown('<div class="main-header">📊 Model Performance</div>', unsafe_allow_html=True)

    # Metrics
    metrics_path = f"outputs/metrics_{dataset_id}.csv"
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path).iloc[0]
        st.subheader("Evaluation Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE",       f"{metrics.get('RMSE', 'N/A'):.2f}")
        col2.metric("MAE",        f"{metrics.get('MAE', 'N/A'):.2f}")
        col3.metric("MAPE",       f"{metrics.get('MAPE (%)', 'N/A'):.1f}%")
        col4.metric("NASA Score", f"{metrics.get('NASA Score', 'N/A'):.1f}")
        st.info("ℹ️ Target RMSE from literature: ~12.5 cycles")
    else:
        st.warning("Run evaluate.py to see metrics")

    st.divider()

    # Loss curve
    loss_img = f"outputs/loss_curve_{dataset_id}.png"
    scatter_img = f"outputs/eval_scatter_{dataset_id}.png"
    per_eng_img = f"outputs/eval_per_engine_{dataset_id}.png"

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(loss_img):
            st.subheader("Training Loss Curve")
            st.image(loss_img, use_column_width=True)
        else:
            st.info("Training loss curve not found. Run train.py.")

    with col_b:
        if os.path.exists(scatter_img):
            st.subheader("Predicted vs Actual (Test Set)")
            st.image(scatter_img, use_column_width=True)
        else:
            st.info("Scatter plot not found. Run evaluate.py.")

    if os.path.exists(per_eng_img):
        st.subheader("Per-Engine RUL Comparison")
        st.image(per_eng_img, use_column_width=True)

    # Detailed results table
    results_path = f"outputs/eval_results_{dataset_id}.csv"
    if os.path.exists(results_path):
        st.subheader("Detailed Results by Engine")
        results = pd.read_csv(results_path)
        st.dataframe(results.style.background_gradient(
            subset=["abs_error"], cmap="YlOrRd"),
            use_container_width=True, height=300
        )


# ──────────────────────────── PAGE 4: SHAP ───────────────────────────
elif page == "🧠 SHAP Explainability":
    st.markdown('<div class="main-header">🧠 SHAP Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("""
    **What is SHAP?**
    SHAP (SHapley Additive exPlanations) uses game theory to explain why the model predicted
    a certain RUL. It answers: *"Which sensors contributed most to this prediction?"*

    - **Positive SHAP** → sensor reading increased the predicted RUL
    - **Negative SHAP** → sensor reading decreased the predicted RUL (higher urgency)
    """)

    shap_img  = f"outputs/shap_summary_{dataset_id}.png"
    imp_path  = f"outputs/feature_importance_{dataset_id}.csv"

    if os.path.exists(shap_img):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.image(shap_img, use_column_width=True)
        with col_b:
            if os.path.exists(imp_path):
                imp_df = pd.read_csv(imp_path)
                st.subheader("Top Features")
                st.dataframe(imp_df[["readable", "mean_abs_shap"]].rename(columns={
                    "readable": "Feature",
                    "mean_abs_shap": "SHAP Value"
                }), use_container_width=True, height=400)
    else:
        st.info("SHAP plots not found. Run `python src/explain.py` first.")


# ──────────────────────────── PAGE 5: Live Prediction ────────────────
elif page == "⚡ Live Prediction":
    st.markdown('<div class="main-header">⚡ Live RUL Prediction</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with sensor readings (at least 30 rows) to get a RUL prediction.")

    if not model_loaded:
        st.error("Model not loaded. Please run the full pipeline first.")
    else:
        tab1, tab2 = st.tabs(["📁 Upload CSV", "✏️ Manual Input"])

        with tab1:
            uploaded = st.file_uploader("Upload sensor CSV (space or comma-separated)",
                                        type=["csv", "txt"])
            if uploaded:
                try:
                    df_upload = pd.read_csv(uploaded, sep=r"\s+|,", engine="python",
                                            header=None)

                    # Try to assign column names
                    from src.bronze_layer import COLS
                    if df_upload.shape[1] == len(COLS):
                        df_upload.columns = COLS
                    elif df_upload.shape[1] == len(COLS) - 2:  # no metadata cols
                        df_upload.columns = COLS[:df_upload.shape[1]]

                    sensor_cols = [c for c in df_upload.columns if c.startswith("s") and c[1:].isdigit()]

                    if len(df_upload) < 10:
                        st.warning("Need at least 10 rows for prediction. More is better.")
                    else:
                        st.success(f"✅ Loaded {len(df_upload)} rows, {len(sensor_cols)} sensors")

                        from src.inference import predict_rul
                        result = predict_rul(df_upload, model, scaler,
                                             meta["feature_cols"], meta["sensor_cols"])

                        col1, col2, col3 = st.columns(3)
                        color = alert_color(result["alert_level"])

                        col1.metric("Predicted RUL", f"{result['predicted_rul']} cycles")
                        col2.metric("Health Index",  f"{result['health_index']}%")
                        col3.markdown(f"""
                        <div style="background:{color}; color:white;
                                    padding:1rem; border-radius:8px; text-align:center;
                                    font-size:1.2rem; font-weight:700; margin-top:0.5rem;">
                            {result['alert_level']}
                        </div>""", unsafe_allow_html=True)

                        # Show gauge
                        gauge_fig = health_gauge(result["health_index"], "Uploaded")
                        st.plotly_chart(gauge_fig, use_container_width=False)

                except Exception as e:
                    st.error(f"Error processing file: {e}")

        with tab2:
            st.markdown("Try prediction with a specific engine from the test set")
            silver_path = f"data/silver/test_{dataset_id}_silver.parquet"
            gold_path   = f"data/gold/test_{dataset_id}_gold.parquet"

            if os.path.exists(gold_path):
                test_gold = pd.read_parquet(gold_path)
                engine_ids = sorted(test_gold["engine_id"].unique())
                sel_eng = st.selectbox("Select test engine", engine_ids)

                if st.button("🔮 Predict RUL", type="primary"):
                    eng_df = test_gold[test_gold["engine_id"] == sel_eng].copy()
                    eng_df = eng_df.sort_values("cycle")

                    from src.inference import predict_rul
                    result = predict_rul(eng_df, model, scaler,
                                         meta["feature_cols"], meta["sensor_cols"])

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted RUL", f"{result['predicted_rul']} cycles")
                    col2.metric("Health Index",  f"{result['health_index']}%")
                    col3.metric("Alert Level", result["alert_level"])

                    gauge_fig = health_gauge(result["health_index"], sel_eng)
                    st.plotly_chart(gauge_fig)

                    # Show sensor history
                    sensor_cols_avail = [c for c in eng_df.columns
                                         if c.startswith("s") and c[1:].isdigit()][:6]
                    if sensor_cols_avail:
                        st.subheader("Sensor History (last 50 cycles)")
                        fig_s = px.line(
                            eng_df.tail(50).melt(
                                id_vars=["cycle"],
                                value_vars=sensor_cols_avail,
                                var_name="sensor",
                                value_name="value"
                            ),
                            x="cycle", y="value", color="sensor",
                            title=f"Engine {sel_eng} — Sensor Readings"
                        )
                        st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("Run the full pipeline first to enable manual prediction.")
