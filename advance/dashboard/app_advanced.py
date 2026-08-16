"""
dashboard/app_advanced.py
--------------------------
ADVANCED PHM DASHBOARD

Additional pages over basic version:
  6. Uncertainty Analysis   — epistemic vs aleatoric breakdown per engine
  7. Model Comparison       — LSTM vs Transformer on same test set
  8. α-λ Performance Map    — 2D heatmap of accuracy vs operating condition

Run:
    streamlit run dashboard/app_advanced.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Advanced PHM — NASA C-MAPSS",
                   page_icon="🔬", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1a1a2e}
.badge-red   {background:#e63946;color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem}
.badge-yellow{background:#f4a261;color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem}
.badge-green {background:#2a9d8f;color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔬 Advanced PHM")
    st.markdown("*CNN-Transformer + Uncertainty*")
    st.divider()
    dataset_id = st.selectbox("Primary Dataset", ["FD001","FD002","FD003","FD004"])
    page = st.radio("Navigation", [
        "🏠 Fleet Overview",
        "🎯 Uncertainty Analysis",
        "📊 Advanced Metrics",
        "⚖️ Model Comparison",
        "📐 Calibration",
        "⚡ Live Prediction",
    ])
    st.divider()
    st.caption("M.Tech Data Engineering")
    st.caption("CNN-Transformer + MC Dropout")
    st.caption("FD001–FD004 Joint Training")


# ──────────────────────────── Fleet Overview ─────────────────────────
if page == "🏠 Fleet Overview":
    st.markdown('<div class="main-header">✈️ Fleet Health — Probabilistic PHM</div>',
                unsafe_allow_html=True)

    adv_results = f"outputs/advanced_results_{dataset_id}.csv"
    fleet_path  = f"outputs/fleet_health_{dataset_id}.csv"

    if os.path.exists(adv_results):
        df = pd.read_csv(adv_results)
        df["engine_id"] = range(1, len(df) + 1)
        df["alert"] = pd.cut(df["pred_rul"],
                              bins=[-1, 20, 50, 999],
                              labels=["CRITICAL", "WARNING", "HEALTHY"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Engines", len(df))
        c2.metric("🔴 Critical (RUL≤20)", (df.alert=="CRITICAL").sum())
        c3.metric("Avg Uncertainty σ",
                  f"{df['uncertainty'].mean():.1f} cycles")
        c4.metric("RMSE",
                  f"{np.sqrt(np.mean((df.pred_rul - df.true_rul)**2)):.2f}")

        st.divider()
        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.subheader("Probabilistic RUL Predictions")
            fig = go.Figure()
            # Sort by engine id
            df_s = df.sort_values("engine_id")
            # Error bars = ±1σ
            fig.add_trace(go.Scatter(
                x=df_s["true_rul"], y=df_s["pred_rul"],
                mode="markers",
                error_y=dict(type="data", array=df_s["uncertainty"].values,
                             visible=True, color="lightgray", thickness=1),
                marker=dict(
                    color=df_s["uncertainty"],
                    colorscale="RdYlGn_r",
                    size=7, showscale=True,
                    colorbar=dict(title="σ (cycles)")
                ),
                text=df_s["engine_id"].apply(lambda x: f"Engine {x}"),
                hovertemplate="<b>%{text}</b><br>Actual RUL: %{x}<br>"
                              "Predicted: %{y:.1f}<extra></extra>"
            ))
            lim = max(df.true_rul.max(), df.pred_rul.max()) + 10
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                          line=dict(color="black", dash="dash"))
            fig.update_layout(height=430, xaxis_title="Actual RUL",
                               yaxis_title="Predicted RUL")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Uncertainty Distribution")
            fig2 = px.histogram(df, x="uncertainty", nbins=20,
                                 color_discrete_sequence=["steelblue"],
                                 labels={"uncertainty": "Prediction σ (cycles)"},
                                 title="How uncertain is the model?")
            fig2.update_layout(height=215)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Error Distribution")
            fig3 = px.histogram(df, x="error", nbins=25,
                                 color_discrete_sequence=["coral"],
                                 labels={"error": "Prediction Error (cycles)"},
                                 title="Error = Predicted − Actual")
            fig3.add_vline(x=0, line_dash="dash", line_color="black")
            fig3.update_layout(height=215)
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Engine Fleet Table — Worst First")
        df_show = df.sort_values("pred_rul")[
            ["engine_id","true_rul","pred_rul","uncertainty","abs_error","alert"]
        ].rename(columns={
            "engine_id":"Engine","true_rul":"Actual RUL",
            "pred_rul":"Predicted RUL","uncertainty":"Uncertainty σ",
            "abs_error":"Abs Error","alert":"Status"
        })
        st.dataframe(df_show, use_container_width=True, height=320)
    else:
        st.warning("Run `python run_pipeline_advanced.py` first.")


# ──────────────────────────── Uncertainty Analysis ───────────────────
elif page == "🎯 Uncertainty Analysis":
    st.markdown('<div class="main-header">🎯 Uncertainty Decomposition</div>',
                unsafe_allow_html=True)

    st.info("""
    **Two types of uncertainty in this model:**

    🔵 **Epistemic uncertainty** — Model doesn't have enough training data for this type of engine.
    Can be reduced with more training data. Captured via Monte Carlo Dropout.

    🟠 **Aleatoric uncertainty** — The sensor data itself is noisy and unpredictable.
    Cannot be reduced even with infinite data. Captured via the Gaussian NLL output head.

    **Total σ = √(epistemic² + aleatoric²)**

    Engines with HIGH total σ should be flagged for physical inspection regardless of RUL prediction.
    """)

    adv_path = f"outputs/advanced_results_{dataset_id}.csv"
    if os.path.exists(adv_path):
        df = pd.read_csv(adv_path)
        df["engine_id"] = range(1, len(df)+1)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Uncertainty vs Prediction Error")
            fig = px.scatter(df, x="uncertainty", y="abs_error",
                              color="true_rul",
                              color_continuous_scale="RdYlGn",
                              labels={
                                  "uncertainty": "Predicted σ (cycles)",
                                  "abs_error": "Absolute Error (cycles)",
                                  "true_rul": "True RUL"
                              },
                              title="Well-calibrated: high σ should correlate with high error")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("95% Confidence Intervals — First 30 Engines")
            df30 = df.head(30).sort_values("true_rul")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(30)), y=df30["true_rul"],
                mode="markers+lines", name="Actual RUL",
                line=dict(color="black", dash="dash"), marker=dict(size=6)
            ))
            fig2.add_trace(go.Scatter(
                x=list(range(30)), y=df30["pred_rul"],
                mode="markers", name="Predicted RUL",
                marker=dict(color="steelblue", size=8),
                error_y=dict(type="data",
                             array=(1.96*df30["uncertainty"]).values,
                             visible=True)
            ))
            fig2.update_layout(height=380,
                                xaxis_title="Engine index (sorted by RUL)",
                                yaxis_title="RUL (cycles)")
            st.plotly_chart(fig2, use_container_width=True)

        # High uncertainty engines
        threshold = df["uncertainty"].quantile(0.75)
        high_unc = df[df["uncertainty"] > threshold].sort_values("uncertainty", ascending=False)
        st.subheader(f"⚠️ High-Uncertainty Engines (σ > {threshold:.1f} cycles)")
        st.dataframe(high_unc[["engine_id","pred_rul","true_rul",
                                 "uncertainty","abs_error"]].head(20),
                     use_container_width=True)
    else:
        st.info("Run the advanced pipeline to generate uncertainty data.")


# ──────────────────────────── Advanced Metrics ───────────────────────
elif page == "📊 Advanced Metrics":
    st.markdown('<div class="main-header">📊 PHD-Level PHM Metrics</div>',
                unsafe_allow_html=True)

    metrics_path = f"outputs/advanced_metrics_{dataset_id}.csv"
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0]

        st.subheader("Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{m.get('RMSE', float('nan')):.2f}",
                   help="Root Mean Square Error on test set")
        c2.metric("MAE",  f"{m.get('MAE', float('nan')):.2f}",
                   help="Mean Absolute Error")
        c3.metric("NASA Score",
                   f"{m.get('NASA_Score', float('nan')):.1f}",
                   help="Asymmetric score — penalizes late predictions more")
        c4.metric("Critical RMSE (RUL≤30)",
                   f"{m.get('Critical_RMSE', float('nan')):.2f}",
                   help="RMSE only on engines near failure — most safety-critical")

        st.divider()
        c5, c6 = st.columns(2)
        c5.metric("Uncertainty ECE",
                   f"{m.get('ECE', float('nan')):.4f}",
                   help="Expected Calibration Error (0 = perfect calibration)")
        c6.metric("Mean Prediction σ",
                   f"{m.get('Mean_Uncertainty', float('nan')):.2f} cycles",
                   help="Average uncertainty across all test predictions")

        st.info("""
        **How to interpret:**
        - **RMSE target**: Literature state-of-art for FD001 is ~12–14 cycles (LSTM baseline ~18)
        - **Critical RMSE**: The most important number for safety — should be < 10
        - **ECE < 0.05**: Well-calibrated uncertainty
        - **NASA Score**: Lower = better. PHM08 winner scored ~250 on FD001
        """)

    # Loss curves
    loss_img = "outputs/advanced_loss_curve.png"
    scatter_img = f"outputs/uncertainty_scatter_{dataset_id}.png"
    calib_img   = f"outputs/calibration_{dataset_id}.png"

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(loss_img):
            st.subheader("Training Loss Curve")
            st.image(loss_img, use_column_width=True)
    with col_b:
        if os.path.exists(scatter_img):
            st.subheader("Probabilistic Scatter")
            st.image(scatter_img, use_column_width=True)

    if os.path.exists(calib_img):
        st.subheader("Calibration Reliability Diagram")
        st.image(calib_img, use_column_width=True)


# ──────────────────────────── Model Comparison ───────────────────────
elif page == "⚖️ Model Comparison":
    st.markdown('<div class="main-header">⚖️ Model Comparison</div>',
                unsafe_allow_html=True)

    st.subheader("Architecture Comparison")

    comparison_data = {
        "Model":           ["LSTM (Basic)",       "CNN-Transformer (Advanced)"],
        "Architecture":    ["2-Layer LSTM",        "CNN + Dual-Attn Transformer"],
        "Training Data":   ["FD001 only (~17K)",   "FD001-4 + Augmented (~100K)"],
        "Uncertainty":     ["None",                "MC Dropout + Gaussian NLL"],
        "Loss Function":   ["MSE",                 "Gaussian NLL"],
        "Training Method": ["Standard",            "Curriculum + Multi-task"],
        "Params":          ["~50K",                "~200K"],
        "Expected RMSE":   ["18–22",               "12–16"],
        "Uncertainty Output": ["❌", "✅"],
        "Calibrated CI":   ["❌", "✅"],
    }
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp.set_index("Model"), use_container_width=True)

    st.divider()
    st.subheader("What Each Advancement Adds")

    improvements = [
        ("Cross-dataset training (FD001-FD004)",
         "6× more training data → model sees more fault modes → better generalization"),
        ("Data augmentation (noise + jitter)",
         "Doubles effective dataset size → reduces overfitting on 100 engines"),
        ("CNN before Transformer",
         "Local pattern detection (adjacent cycle spikes) that Transformer alone misses"),
        ("Dual-axis attention (sensor + time)",
         "Model learns WHICH sensors and WHICH timesteps matter — not equal weight"),
        ("Curriculum learning",
         "Starts on healthy engines → progressively adds near-failure samples"),
        ("Gaussian NLL loss",
         "Model learns its own uncertainty → knows when to say 'I'm not sure'"),
        ("MC Dropout at inference",
         "50 stochastic forward passes → epistemic + aleatoric uncertainty bands"),
        ("α-λ metric",
         "Evaluates: 'At 60% of engine life, are predictions within ±20%?' → PHM standard"),
        ("Calibration (ECE)",
         "Verifies uncertainty intervals are statistically honest — critical for safety"),
    ]

    for title, explanation in improvements:
        with st.expander(f"📌 {title}"):
            st.write(explanation)


# ──────────────────────────── Calibration ────────────────────────────
elif page == "📐 Calibration":
    st.markdown('<div class="main-header">📐 Uncertainty Calibration</div>',
                unsafe_allow_html=True)

    calib_img = f"outputs/calibration_{dataset_id}.png"
    if os.path.exists(calib_img):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.image(calib_img, use_column_width=True)
        with col_b:
            st.markdown("""
            **How to read this:**

            The diagonal line = perfect calibration.

            If the blue line is **below** the diagonal:
            - Model is **overconfident**
            - 95% CI actually contains truth only 80% of time
            - Dangerous in safety applications!

            If the blue line is **above** the diagonal:
            - Model is **underconfident**
            - 95% CI is actually wider than needed
            - Conservative but safe.

            **Goal:** blue line follows the diagonal closely.

            **ECE (Expected Calibration Error):**
            Average distance between blue line and diagonal.
            ECE < 0.05 is considered well-calibrated.
            """)
    else:
        st.info("Run the advanced pipeline to generate calibration data.")


# ──────────────────────────── Live Prediction ────────────────────────
elif page == "⚡ Live Prediction":
    st.markdown('<div class="main-header">⚡ Live Probabilistic Prediction</div>',
                unsafe_allow_html=True)
    st.markdown("Upload sensor CSV → get RUL prediction **with uncertainty intervals**")

    @st.cache_resource
    def load_adv_model(dataset_id):
        try:
            from src.models.transformer_rul import CNNTransformerRUL
            meta = joblib.load(f"data/gold/feature_meta_{dataset_id}.pkl")
            model = CNNTransformerRUL(n_features=len(meta["feature_cols"]),
                                      d_model=64, n_heads=4,
                                      n_transformer_layers=2, dropout=0.15)
            model_path = "models/transformer_best.pth"
            if not os.path.exists(model_path):
                model_path = f"models/lstm_{dataset_id}_best.pth"
                from src.model import LSTMModel
                model = LSTMModel(input_size=len(meta["feature_cols"]))
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            scaler = joblib.load(f"models/scaler_{dataset_id}.pkl")
            return model, scaler, meta, True
        except Exception as e:
            return None, None, None, False

    model, scaler, meta, loaded = load_adv_model(dataset_id)

    if not loaded:
        st.error("Model not loaded. Run the pipeline first.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            uploaded = st.file_uploader("Upload sensor readings CSV", type=["csv","txt"])
        with col_b:
            mc_n = st.slider("MC Dropout samples", 10, 100, 50,
                              help="More samples = better uncertainty estimate but slower")

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded, sep=r"\s+|,", engine="python", header=None)
                from src.bronze_layer import COLS
                if df_up.shape[1] == len(COLS):
                    df_up.columns = COLS

                st.success(f"✅ Loaded {len(df_up)} rows")

                from src.inference import predict_rul
                from src.models.uncertainty import mc_dropout_predict
                from src.gold_layer import add_lag_features, add_rolling_features, WINDOW_SIZE

                sensor_cols = meta["sensor_cols"]
                feature_cols = meta["feature_cols"]
                ZERO_VAR = ["s1","s5","s6","s10","s16","s18","s19"]
                df_up = df_up.drop(columns=[c for c in ZERO_VAR if c in df_up.columns],
                                   errors="ignore")

                avail = [c for c in sensor_cols if c in df_up.columns]
                df_norm = df_up.copy()
                df_norm[avail] = scaler.transform(df_up[avail])
                df_norm = add_lag_features(df_norm, avail)
                df_norm = add_rolling_features(df_norm, avail)
                df_norm = df_norm.fillna(0)

                avail_f = [c for c in feature_cols if c in df_norm.columns]
                data = df_norm[avail_f].values[-WINDOW_SIZE:]
                if len(data) < WINDOW_SIZE:
                    data = np.vstack([np.zeros((WINDOW_SIZE-len(data), data.shape[1])), data])
                if data.shape[1] < len(feature_cols):
                    data = np.hstack([data, np.zeros((len(data), len(feature_cols)-data.shape[1]))])

                X_t = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                result = mc_dropout_predict(model, X_t, n_samples=mc_n)

                rul_mean  = max(0.0, float(result["rul_mean"]))
                total_std = float(result["total_std"])
                lower_95  = max(0.0, float(result["rul_lower_95"]))
                upper_95  = float(result["rul_upper_95"])

                alert = ("CRITICAL" if rul_mean <= 20 else
                         "WARNING"  if rul_mean <= 50 else "HEALTHY")
                hi    = min(100.0, rul_mean / 125 * 100)

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Predicted RUL", f"{rul_mean:.1f} cycles")
                c2.metric("Uncertainty (1σ)", f"±{total_std:.1f} cycles")
                c3.metric("95% CI",
                           f"[{lower_95:.0f}, {upper_95:.0f}]")
                c4.metric("Health Index", f"{hi:.0f}%")

                # Gauge
                color = {"CRITICAL":"#e63946","WARNING":"#f4a261","HEALTHY":"#2a9d8f"}[alert]
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=hi,
                    title={"text": f"Health Index — {alert}"},
                    gauge={
                        "axis": {"range": [0,100]},
                        "bar":  {"color": color},
                        "steps": [
                            {"range":[0,30],"color":"#fde8e8"},
                            {"range":[30,60],"color":"#fff3e0"},
                            {"range":[60,100],"color":"#e8f5e9"}
                        ]
                    },
                    number={"suffix":"%"}
                ))
                gauge.update_layout(height=260)
                st.plotly_chart(gauge, use_container_width=False)

                # MC samples distribution
                mc_arr = result["all_samples"].flatten()
                fig_mc = px.histogram(mc_arr, nbins=20,
                                       title=f"MC Dropout Distribution ({mc_n} samples)",
                                       labels={"value":"Predicted RUL (cycles)"},
                                       color_discrete_sequence=["steelblue"])
                fig_mc.add_vline(x=rul_mean, line_dash="dash", line_color="red",
                                  annotation_text=f"Mean={rul_mean:.1f}")
                st.plotly_chart(fig_mc, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
