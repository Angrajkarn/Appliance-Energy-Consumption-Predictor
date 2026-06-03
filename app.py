"""
======================================================================
  Appliance Energy Consumption Prediction
  Interactive Streamlit Dashboard — app.py
  
  Launch: streamlit run app.py
======================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    color: #E0E0FF;
  }

  /* ── Hero Header ── */
  .hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #A855F7 50%, #EC4899 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(108,99,255,0.4);
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
  }
  .hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    text-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin: 0;
    position: relative;
  }
  .hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
    position: relative;
  }

  /* ── Metric Cards ── */
  .metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(168,85,247,0.1) 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(108,99,255,0.3);
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #6C63FF;
  }
  .metric-label {
    font-size: 0.85rem;
    color: #A0A0C0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
  }
  .metric-delta {
    font-size: 0.85rem;
    color: #00C9A7;
    font-weight: 600;
  }

  /* ── Section Headers ── */
  .section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E0E0FF;
    border-left: 4px solid #6C63FF;
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* ── Prediction Box ── */
  .prediction-box {
    background: linear-gradient(135deg, #00C9A7 0%, #6C63FF 100%);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0,201,167,0.3);
    animation: glow 2s ease-in-out infinite alternate;
  }
  @keyframes glow {
    from { box-shadow: 0 15px 40px rgba(0,201,167,0.3); }
    to   { box-shadow: 0 15px 60px rgba(108,99,255,0.5); }
  }
  .prediction-value {
    font-size: 3.5rem;
    font-weight: 900;
    color: white;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .prediction-label {
    font-size: 1rem;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
    border-right: 1px solid rgba(108,99,255,0.2);
  }

  /* ── Plotly chart background ── */
  .js-plotly-plot .plotly .bg {
    fill: transparent !important;
  }

  /* ── Info box ── */
  .info-box {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
  }

  /* ── Status badge ── */
  .badge {
    display: inline-block;
    background: linear-gradient(90deg, #6C63FF, #A855F7);
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .badge-green { background: linear-gradient(90deg, #00C9A7, #22D3EE); }
  .badge-orange { background: linear-gradient(90deg, #F7931E, #F43F5E); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS & THEME
# ─────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(15,15,26,0)",
    plot_bgcolor="rgba(26,26,46,0.7)",
    font=dict(family="Inter", color="#E0E0FF"),
    xaxis=dict(gridcolor="#2D2D4E", linecolor="#2D2D4E"),
    yaxis=dict(gridcolor="#2D2D4E", linecolor="#2D2D4E"),
    margin=dict(l=40, r=20, t=50, b=40),
)
COLORS = ["#6C63FF", "#F7931E", "#00C9A7", "#FF6B6B", "#A855F7", "#22D3EE", "#F43F5E", "#84CC16", "#FB923C"]

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_engineer(path="energydata_complete.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)
    df.drop(columns=["rv1", "rv2"], inplace=True, errors="ignore")

    # Temporal
    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["time_of_day"] = df["hour"].apply(lambda h: 0 if h < 6 else (1 if h < 12 else (2 if h < 18 else 3)))

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 144]:
        df[f"lag_{lag}"] = df["Appliances"].shift(lag)

    # Rolling stats
    for w in [3, 6, 12, 24, 48]:
        df[f"rolling_mean_{w}"] = df["Appliances"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["Appliances"].shift(1).rolling(w).std()
        df[f"rolling_max_{w}"]  = df["Appliances"].shift(1).rolling(w).max()
        df[f"rolling_min_{w}"]  = df["Appliances"].shift(1).rolling(w).min()

    # Differentials
    for c in [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]:
        df[f"dT_{c}_out"] = df[c] - df["T_out"]
        df[f"dT_{c}_T1"]  = df[c] - df["T1"]
    for c in [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()]:
        df[f"dRH_{c}_out"] = df[c] - df["RH_out"]

    # Interactions
    df["lights_x_hour"]    = df["lights"] * df["hour"]
    df["T1_RH1_interact"]  = df["T1"] * df["RH_1"]
    df["avg_indoor_temp"]  = df[[c for c in df.columns if c.startswith("T") and c[1:].isdigit() and int(c[1:]) <= 9]].mean(axis=1)
    df["avg_indoor_humid"] = df[[c for c in df.columns if c.startswith("RH_") and c[3:].isdigit() and int(c[3:]) <= 9]].mean(axis=1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data(show_spinner=False)
def load_raw(path="energydata_complete.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df.sort_values("date").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    model_files = {
        "Stacking Ensemble": "models/stacking_regressor.pkl",
        "LightGBM (Tuned)": "models/lightgbm_tuned.pkl",
        "LightGBM":         "models/lightgbm.pkl",
        "XGBoost":          "models/xgboost.pkl",
        "Random Forest":    "models/random_forest.pkl",
        "CatBoost":         "models/catboost.pkl",
        "MLP Neural Net":   "models/mlp.pkl",
        "SVR":              "models/svr.pkl",
        "Ridge":            "models/ridge.pkl",
        "Lasso":            "models/lasso.pkl",
        "Linear Regression":"models/linear_regression.pkl",
        "Decision Tree":    "models/decision_tree.pkl",
    }
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception:
                pass
    scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
    return loaded, scaler

def load_results():
    if os.path.exists("results/model_comparison.csv"):
        return pd.read_csv("results/model_comparison.csv")
    return None

def load_feature_importance():
    if os.path.exists("results/feature_importance.csv"):
        return pd.read_csv("results/feature_importance.csv", index_col=0)
    return None

# ─────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">⚡ Appliance Energy Consumption Predictor</div>
  <div class="hero-subtitle">Advanced Machine Learning Dashboard · Household IoT Sensor Analysis · Real-time Predictions</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio("", [
        "📊 Overview & EDA",
        "🔍 Feature Analysis",
        "🤖 Model Performance",
        "🎯 Live Prediction",
        "🚨 Anomaly Explorer",
        "💡 Optimization Simulator",
        "📈 Time-Series Deep Dive",
        "🧠 Model Explainability"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_raw = st.checkbox("Show raw data preview", False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#6060A0; line-height:1.6;'>
    <b>Dataset Info</b><br>
    📁 energydata_complete.csv<br>
    🕐 10-min intervals<br>
    📅 Jan–May 2016<br>
    📊 19,735 rows · 29 features<br>
    🎯 Target: Appliances (Wh)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    pipeline_ran = os.path.exists("models/lightgbm_tuned.pkl")
    if pipeline_ran:
        st.markdown('<span class="badge badge-green">✅ Models Trained</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">⚠️ Run Pipeline First</span>', unsafe_allow_html=True)
        st.code("python run_pipeline.py", language="bash")

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────
DATA_PATH = "energydata_complete.csv"
if not os.path.exists(DATA_PATH):
    st.error("❌ `energydata_complete.csv` not found. Please place it in the same directory as `app.py`.")
    st.stop()

with st.spinner("Loading and engineering features..."):
    df_raw = load_raw(DATA_PATH)
    df_eng = load_and_engineer(DATA_PATH)

FEATURE_COLS = [c for c in df_eng.columns if c not in ["Appliances", "date"]]
X_all = df_eng[FEATURE_COLS]
y_all = df_eng["Appliances"]

if show_raw:
    st.dataframe(df_raw.head(50), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW & EDA
# ─────────────────────────────────────────────────────────────────────
if "Overview" in page:
    # ── KPI Cards ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, f"{len(df_raw):,}", "Total Samples", "10-min intervals"),
        (c2, f"{df_raw['Appliances'].mean():.0f} Wh", "Mean Energy", f"Std: {df_raw['Appliances'].std():.0f} Wh"),
        (c3, f"{df_raw['Appliances'].max():.0f} Wh", "Peak Consumption", f"Min: {df_raw['Appliances'].min():.0f} Wh"),
        (c4, "29", "Original Features", "After cleaning: ~90+"),
        (c5, f"{(df_raw['date'].max() - df_raw['date'].min()).days}", "Days Covered", "Jan–May 2016"),
    ]
    for col, val, label, delta in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
              <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time-Series Plot ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Energy Consumption Over Time</div>', unsafe_allow_html=True)
    date_range = st.slider("Select date range (days from start)",
                            0, len(df_raw) // 144,
                            (0, min(14, len(df_raw) // 144)))
    start_idx = date_range[0] * 144
    end_idx   = date_range[1] * 144
    sample_df = df_raw.iloc[start_idx:end_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_df["date"], y=sample_df["Appliances"],
        mode="lines", name="Appliances",
        line=dict(color="#6C63FF", width=1.5),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=sample_df["date"], y=sample_df["lights"],
        mode="lines", name="Lights",
        line=dict(color="#F7931E", width=1.2),
        fill="tozeroy", fillcolor="rgba(247,147,30,0.08)"
    ))
    fig.update_layout(title="Appliances & Lights Energy Consumption", **PLOTLY_THEME,
                      legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2D2D4E"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Temporal Patterns ────────────────────────────────────────────
    st.markdown('<div class="section-header">🕐 Temporal Consumption Patterns</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        hourly = df_raw.copy()
        hourly["hour"] = hourly["date"].dt.hour
        hourly_stats = hourly.groupby("hour")["Appliances"].agg(["mean", "std"]).reset_index()
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=hourly_stats["hour"], y=hourly_stats["mean"],
            name="Mean", marker_color="#6C63FF", opacity=0.85,
            error_y=dict(type="data", array=hourly_stats["std"] * 0.5, visible=True,
                          color="rgba(255,255,255,0.4)")
        ))
        fig_h.update_layout(title="Hourly Average Consumption", xaxis_title="Hour of Day",
                            yaxis_title="Energy (Wh)", **PLOTLY_THEME)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        daily = df_raw.copy()
        daily["dow"] = daily["date"].dt.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_stats = daily.groupby("dow")["Appliances"].mean().reset_index()
        bar_colors_d = ["#FF6B6B" if i >= 5 else "#00C9A7" for i in range(7)]
        fig_d = go.Figure(go.Bar(
            x=day_names, y=daily_stats["Appliances"],
            marker_color=bar_colors_d, opacity=0.85
        ))
        fig_d.update_layout(title="Average Consumption by Day", xaxis_title="Day",
                            yaxis_title="Energy (Wh)", **PLOTLY_THEME)
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Monthly Heatmap ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🗓️ Hourly × Day-of-Week Heatmap</div>', unsafe_allow_html=True)
    df_raw["hour"]  = df_raw["date"].dt.hour
    df_raw["dow"]   = df_raw["date"].dt.dayofweek
    pivot = df_raw.pivot_table(values="Appliances", index="hour", columns="dow", aggfunc="mean")
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_heat = px.imshow(pivot, color_continuous_scale="Plasma", aspect="auto",
                          labels=dict(x="Day of Week", y="Hour of Day", color="Wh"),
                          title="Avg Consumption Heatmap (Hour × Day)")
    fig_heat.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Distribution ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Target Distribution</div>', unsafe_allow_html=True)
    col_x, col_y = st.columns(2)
    with col_x:
        fig_dist = px.histogram(df_raw, x="Appliances", nbins=80,
                                 color_discrete_sequence=["#6C63FF"],
                                 title="Appliance Energy Distribution",
                                 labels={"Appliances": "Energy (Wh)"})
        fig_dist.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_y:
        df_raw["log_Appliances"] = np.log1p(df_raw["Appliances"])
        fig_logdist = px.histogram(df_raw, x="log_Appliances", nbins=80,
                                    color_discrete_sequence=["#00C9A7"],
                                    title="Log-Transformed Distribution",
                                    labels={"log_Appliances": "log(Energy+1)"})
        fig_logdist.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_logdist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE: FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────
elif "Feature" in page:
    st.markdown('<div class="section-header">🔍 Feature Correlation with Target</div>', unsafe_allow_html=True)
    numeric_df = df_raw.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False).head(20)

    fig_corr = go.Figure(go.Bar(
        y=corr.index, x=corr.values, orientation="h",
        marker=dict(
            color=corr.values,
            colorscale="RdYlGn",
            cmin=-0.5, cmax=0.5,
            showscale=True
        )
    ))
    fig_corr.update_layout(title="Top 20 Feature Correlations with Appliances",
                            xaxis_title="Pearson Correlation", yaxis_title="",
                            **PLOTLY_THEME, height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Pairplot substitute ──────────────────────────────────────────
    st.markdown('<div class="section-header">🌡️ Temperature vs Energy</div>', unsafe_allow_html=True)
    temp_cols_sel = ["T1", "T2", "T3", "T6", "T_out"]
    col_sel = st.selectbox("Select temperature sensor:", temp_cols_sel)

    fig_scatter = px.scatter(df_raw.sample(2000), x=col_sel, y="Appliances",
                              color="hour" if "hour" in df_raw.columns else None,
                              color_continuous_scale="Plasma", opacity=0.6,
                              labels={col_sel: f"{col_sel} (°C)", "Appliances": "Energy (Wh)"},
                              title=f"{col_sel} vs Appliance Energy Consumption",
                              trendline="ols")
    fig_scatter.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Correlation Matrix ────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Correlation Matrix (Temperature & Humidity)</div>', unsafe_allow_html=True)
    sel_cols = [c for c in df_raw.columns if (c.startswith("T") and c[1:].isdigit()) or
                (c.startswith("RH_") and c[3:].isdigit()) or c == "Appliances"] + ["lights"]
    corr_matrix = df_raw[sel_cols].corr()
    fig_cm = px.imshow(corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", title="Temperature & Humidity Correlation Matrix")
    fig_cm.update_layout(**PLOTLY_THEME, height=600)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Box plots ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📦 Energy Distribution by Time of Day</div>', unsafe_allow_html=True)
    df_raw["time_of_day"] = df_raw["date"].dt.hour.apply(
        lambda h: "Night (0-5)" if h < 6 else ("Morning (6-11)" if h < 12 else
                  ("Afternoon (12-17)" if h < 18 else "Evening (18-23)"))
    )
    fig_box = px.box(df_raw, x="time_of_day", y="Appliances",
                      color="time_of_day",
                      color_discrete_sequence=["#6C63FF", "#F7931E", "#00C9A7", "#FF6B6B"],
                      category_orders={"time_of_day": ["Night (0-5)", "Morning (6-11)", "Afternoon (12-17)", "Evening (18-23)"]},
                      title="Appliance Energy by Time of Day",
                      labels={"Appliances": "Energy (Wh)", "time_of_day": "Period"})
    fig_box.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
elif "Model Performance" in page:
    st.markdown('<div class="section-header">🤖 Model Comparison Dashboard</div>', unsafe_allow_html=True)

    results_df = load_results()

    if results_df is None:
        st.warning("⚠️ Model results not found. Run `python run_pipeline.py` first.")
    else:
        # ── Summary Table ──────────────────────────────────────────
        st.markdown("**📋 All Model Results**")
        styled = results_df.sort_values("R²", ascending=False).reset_index(drop=True)
        st.dataframe(
            styled.style
            .background_gradient(subset=["R²"], cmap="Greens")
            .background_gradient(subset=["RMSE", "MAE"], cmap="Reds_r")
            .format({"R²": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE(%)": "{:.2f}"}),
            use_container_width=True
        )

        # ── Radar Chart ────────────────────────────────────────────
        st.markdown('<div class="section-header">🕸️ Multi-Metric Radar Comparison</div>', unsafe_allow_html=True)
        top_models = results_df.nlargest(5, "R²")
        categories = ["R²", "1/MAE (norm)", "1/RMSE (norm)"]

        fig_radar = go.Figure()
        norm_mae  = 1 / (results_df["MAE"] + 1e-9)
        norm_rmse = 1 / (results_df["RMSE"] + 1e-9)
        max_r2   = results_df["R²"].max()
        max_nmae = norm_mae.max()
        max_nrmse= norm_rmse.max()

        for i, (_, row) in enumerate(top_models.iterrows()):
            vals_row = results_df[results_df["Model"] == row["Model"]]
            r2_n   = row["R²"] / max_r2
            mae_n  = (1 / (row["MAE"]  + 1e-9)) / max_nmae
            rmse_n = (1 / (row["RMSE"] + 1e-9)) / max_nrmse
            values = [r2_n, mae_n, rmse_n, r2_n]
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories + [categories[0]],
                fill="toself", name=row["Model"],
                line=dict(color=COLORS[i % len(COLORS)]),
                fillcolor=COLORS[i % len(COLORS)].replace("#", "rgba(") + ",0.15)"
                    if "#" not in COLORS[i % len(COLORS)] else COLORS[i % len(COLORS)]
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(26,26,46,0.7)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2D2D4E", color="#6060A0"),
                angularaxis=dict(gridcolor="#2D2D4E", color=PALETTE["text"] if "PALETTE" in dir() else "#E0E0FF")
            ),
            paper_bgcolor="rgba(15,15,26,0)",
            font=dict(family="Inter", color="#E0E0FF"),
            title="Top 5 Models — Normalized Multi-Metric Comparison",
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Bar Charts ─────────────────────────────────────────────
        col_r2, col_mae, col_rmse = st.columns(3)
        for col, metric, cscale, ascending in [
            (col_r2,   "R²",   "Blues",  False),
            (col_mae,  "MAE",  "Reds_r", True),
            (col_rmse, "RMSE", "Oranges_r", True)
        ]:
            with col:
                sorted_res = results_df.sort_values(metric, ascending=ascending)
                fig_b = px.bar(sorted_res, x=metric, y="Model", orientation="h",
                                color=metric, color_continuous_scale=cscale,
                                title=metric, labels={"Model": ""})
                fig_b.update_layout(**PLOTLY_THEME, height=380, showlegend=False,
                                     coloraxis_showscale=False)
                st.plotly_chart(fig_b, use_container_width=True)

    # ── Feature Importance ─────────────────────────────────────────
    fi = load_feature_importance()
    if fi is not None:
        st.markdown('<div class="section-header">🎯 Feature Importance (Best Model)</div>', unsafe_allow_html=True)
        fi_sorted = fi.sort_values("importance", ascending=False).head(25)
        fig_fi = px.bar(fi_sorted.reset_index(), x="importance", y="index",
                         orientation="h", color="importance",
                         color_continuous_scale="Viridis",
                         title="Top 25 Most Important Features",
                         labels={"index": "Feature", "importance": "Importance Score"})
        fig_fi.update_layout(**PLOTLY_THEME, height=700, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE: LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────
elif "Prediction" in page:
    st.markdown('<div class="section-header">🎯 Real-Time Energy Consumption Predictor</div>', unsafe_allow_html=True)

    models, scaler = load_models()

    if not models:
        st.warning("⚠️ No trained models found. Run `python run_pipeline.py` first.")
    else:
        # Model selector
        model_name = st.selectbox("🤖 Select Model", list(models.keys()))
        selected_model = models[model_name]

        st.markdown("---")
        st.markdown("#### 🌡️ Set Environmental Conditions")
        st.markdown('<div class="info-box">Adjust sliders to match your household conditions. The model will instantly predict appliance energy consumption.</div>', unsafe_allow_html=True)

        # ── Input Sliders ──────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏠 Indoor Temperatures (°C)**")
            T1  = st.slider("T1 — Kitchen",      15.0, 30.0, 20.0, 0.1)
            T2  = st.slider("T2 — Living Room",  15.0, 30.0, 21.0, 0.1)
            T3  = st.slider("T3 — Laundry Room", 15.0, 30.0, 20.5, 0.1)
            T4  = st.slider("T4 — Office",       15.0, 30.0, 20.0, 0.1)
            T5  = st.slider("T5 — Bathroom",     15.0, 30.0, 22.0, 0.1)
            T6  = st.slider("T6 — Outside (N)",  -5.0, 25.0, 7.0,  0.1)
            T7  = st.slider("T7 — Ironing Room", 15.0, 30.0, 20.0, 0.1)
            T8  = st.slider("T8 — Teen Room",    15.0, 30.0, 21.0, 0.1)
            T9  = st.slider("T9 — Parents Room", 15.0, 30.0, 20.5, 0.1)
            lights = st.slider("💡 Lights (Wh)", 0, 70, 0)

        with col2:
            st.markdown("**💧 Indoor Humidity (%)**")
            RH1 = st.slider("RH_1 — Kitchen",      20.0, 70.0, 47.0, 0.5)
            RH2 = st.slider("RH_2 — Living Room",  20.0, 70.0, 46.0, 0.5)
            RH3 = st.slider("RH_3 — Laundry",      20.0, 70.0, 53.0, 0.5)
            RH4 = st.slider("RH_4 — Office",       20.0, 70.0, 46.0, 0.5)
            RH5 = st.slider("RH_5 — Bathroom",     20.0, 70.0, 50.0, 0.5)
            RH6 = st.slider("RH_6 — Outside (N)",  1.0,  100.0,99.0, 0.5)
            RH7 = st.slider("RH_7 — Ironing",      20.0, 70.0, 45.0, 0.5)
            RH8 = st.slider("RH_8 — Teen Room",    20.0, 70.0, 44.0, 0.5)
            RH9 = st.slider("RH_9 — Parents Room", 20.0, 70.0, 45.0, 0.5)

        with col3:
            st.markdown("**🌤️ Weather Conditions**")
            To         = st.slider("To — Outdoor Temp (°C)",    -5.0, 25.0, 6.0, 0.1)
            Pressure   = st.slider("Pressure (mm Hg)",          729.0, 772.0, 755.0, 0.5)
            RH_out     = st.slider("RH_out — Outdoor Humidity", 1.0, 100.0, 92.0, 0.5)
            Wind_speed = st.slider("Wind Speed (m/s)",          0.0, 15.0, 4.0, 0.1)
            Visibility = st.slider("Visibility (km)",           1.0, 66.0, 38.0, 0.5)
            Tdewpoint  = st.slider("Dew Point (°C)",            -5.0, 15.0, 4.7, 0.1)

            st.markdown("**⏰ Time**")
            hour       = st.slider("Hour of Day",  0, 23, 12)
            day_of_week= st.slider("Day of Week (0=Mon)", 0, 6, 1)
            month      = st.slider("Month",        1, 12, 3)

            st.markdown("**📊 Lag Values (Wh)**")
            lag1  = st.number_input("Energy 10 min ago",  0, 1000, 60)
            lag6  = st.number_input("Energy 1 hour ago",  0, 1000, 70)
            lag12 = st.number_input("Energy 2 hours ago", 0, 1000, 65)

        # ── Build feature vector ───────────────────────────────────
        def build_feature_vector():
            week        = (month - 1) * 4 + 2
            day_of_year = month * 30
            is_weekend  = int(day_of_week >= 5)
            quarter     = (month - 1) // 3 + 1
            time_of_day = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))

            rolling_mean = (lag1 + lag6 + lag12) / 3
            rolling_std  = np.std([lag1, lag6, lag12])

            row = {
                "lights": lights, "T1": T1, "RH_1": RH1, "T2": T2, "RH_2": RH2,
                "T3": T3, "RH_3": RH3, "T4": T4, "RH_4": RH4, "T5": T5, "RH_5": RH5,
                "T6": T6, "RH_6": RH6, "T7": T7, "RH_7": RH7, "T8": T8, "RH_8": RH8,
                "T9": T9, "RH_9": RH9, "T_out": To, "Press_mm_hg": Pressure, "RH_out": RH_out,
                "Windspeed": Wind_speed, "Visibility": Visibility, "Tdewpoint": Tdewpoint,
                "hour": hour, "day_of_week": day_of_week, "month": month,
                "week": week, "day_of_year": day_of_year, "is_weekend": is_weekend,
                "quarter": quarter, "time_of_day": time_of_day,
                "lag_1": lag1, "lag_2": lag1 * 0.95, "lag_3": lag1 * 0.9,
                "lag_6": lag6, "lag_12": lag12, "lag_24": lag12 * 0.95,
                "lag_48": lag12 * 0.9, "lag_144": rolling_mean,
                "rolling_mean_3": rolling_mean, "rolling_std_3": rolling_std,
                "rolling_max_3": max(lag1, lag6, lag12), "rolling_min_3": min(lag1, lag6, lag12),
                "rolling_mean_6": rolling_mean, "rolling_std_6": rolling_std,
                "rolling_max_6": max(lag1, lag6, lag12), "rolling_min_6": min(lag1, lag6, lag12),
                "rolling_mean_12": rolling_mean, "rolling_std_12": rolling_std,
                "rolling_max_12": max(lag1, lag6, lag12), "rolling_min_12": min(lag1, lag6, lag12),
                "rolling_mean_24": rolling_mean, "rolling_std_24": rolling_std,
                "rolling_max_24": max(lag1, lag6, lag12), "rolling_min_24": min(lag1, lag6, lag12),
                "rolling_mean_48": rolling_mean, "rolling_std_48": rolling_std,
                "rolling_max_48": max(lag1, lag6, lag12), "rolling_min_48": min(lag1, lag6, lag12),
            }

            # Temperature differentials
            for i, (tc, tv) in enumerate(zip(
                ["T1","T2","T3","T4","T5","T6","T7","T8","T9"],
                [T1, T2, T3, T4, T5, T6, T7, T8, T9]
            ), 1):
                row[f"dT_{tc}_out"] = tv - To
                row[f"dT_{tc}_T1"]  = tv - T1

            # Humidity differentials
            for rc, rv in zip(
                ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"],
                [RH1, RH2, RH3, RH4, RH5, RH6, RH7, RH8, RH9]
            ):
                row[f"dRH_{rc}_out"] = rv - RH_out

            # Interactions
            row["lights_x_hour"]    = lights * hour
            row["T1_RH1_interact"]  = T1 * RH1
            row["avg_indoor_temp"]  = np.mean([T1,T2,T3,T4,T5,T6,T7,T8,T9])
            row["avg_indoor_humid"] = np.mean([RH1,RH2,RH3,RH4,RH5,RH6,RH7,RH8,RH9])

            # Build DataFrame aligned to training features
            row_df = pd.DataFrame([row])
            for col in FEATURE_COLS:
                if col not in row_df.columns:
                    row_df[col] = 0.0
            return row_df[FEATURE_COLS]

        # ── Predict ────────────────────────────────────────────────
        input_df = build_feature_vector()

        linear_models = ["Linear Regression", "Ridge", "Lasso", "SVR", "MLP Neural Net"]
        if any(m in model_name for m in linear_models) and scaler is not None:
            input_arr = scaler.transform(input_df)
            prediction = selected_model.predict(input_arr)[0]
        else:
            prediction = selected_model.predict(input_df)[0]

        prediction = max(0, prediction)
        daily_est  = prediction * 6 * 24  # 6 readings/hr × 24 hrs → Wh
        cost_est   = daily_est / 1000 * 0.12  # $0.12 per kWh

        # ── Display prediction ─────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        with c1:
            st.markdown(f"""
            <div class="prediction-box">
              <div class="prediction-label">⚡ Predicted Appliance Energy</div>
              <div class="prediction-value">{prediction:.1f} <span style='font-size:1.8rem;'>Wh</span></div>
              <div class="prediction-label" style='margin-top:0.5rem;'>per 10-minute interval</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="height:100%">
              <div class="metric-value" style="color:#F7931E">{daily_est/1000:.2f} kWh</div>
              <div class="metric-label">Estimated Daily Usage</div>
              <div class="metric-delta">at this consumption rate</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="height:100%">
              <div class="metric-value" style="color:#00C9A7">${cost_est:.3f}</div>
              <div class="metric-label">Estimated Daily Cost</div>
              <div class="metric-delta">@ $0.12 / kWh</div>
            </div>""", unsafe_allow_html=True)

        # ── Gauge chart ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": df_raw["Appliances"].mean(), "valueformat": ".1f",
                   "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, df_raw["Appliances"].quantile(0.99)], "tickwidth": 1},
                "bar":  {"color": "#6C63FF"},
                "steps": [
                    {"range": [0,   100], "color": "rgba(0,201,167,0.2)"},
                    {"range": [100, 300], "color": "rgba(247,147,30,0.2)"},
                    {"range": [300, 1060],"color": "rgba(255,107,107,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#F7931E", "width": 3},
                    "thickness": 0.75,
                    "value": df_raw["Appliances"].mean()
                },
                "bgcolor": "rgba(26,26,46,0.7)",
                "bordercolor": "#2D2D4E"
            },
            title={"text": "Energy Consumption Gauge (Wh per 10-min)", "font": {"size": 14}},
            number={"suffix": " Wh", "font": {"size": 40, "color": "#6C63FF"}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(15,15,26,0)", font=dict(color="#E0E0FF", family="Inter"), height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE: ANOMALY EXPLORER
# ─────────────────────────────────────────────────────────────────────
elif "Anomaly Explorer" in page:
    st.markdown('<div class="section-header">🚨 Unsupervised Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Using an <b>Isolation Forest</b> algorithm to detect unusual energy consumption spikes based on multivariate sensor relationships.</div>', unsafe_allow_html=True)

    if os.path.exists("models/isolation_forest.pkl"):
        iso_forest = joblib.load("models/isolation_forest.pkl")
        
        # We need to scale df_eng just like we scaled X_train
        _, scaler = load_models()
        if scaler:
            X_all_sc = scaler.transform(X_all)
            preds = iso_forest.predict(X_all_sc)
            
            # Map predictions
            df_anomaly = df_raw.copy()
            df_anomaly["is_anomaly"] = preds == -1
            
            total_anomalies = df_anomaly["is_anomaly"].sum()
            st.markdown(f"**Found {total_anomalies} anomalous periods** ({total_anomalies/len(df_anomaly):.1%}) in the entire dataset.")
            
            # Scatter Plot of anomalies
            fig = px.scatter(df_anomaly, x="date", y="Appliances", color="is_anomaly",
                             color_discrete_map={False: "#6C63FF", True: "#FF6B6B"},
                             title="Energy Spikes flagged as Anomalies (Red)",
                             labels={"date": "Date", "Appliances": "Energy (Wh)", "is_anomaly": "Anomaly?"})
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution comparison
            col_a, col_b = st.columns(2)
            with col_a:
                fig_box1 = px.box(df_anomaly, x="is_anomaly", y="Appliances", color="is_anomaly",
                                  color_discrete_map={False: "#6C63FF", True: "#FF6B6B"},
                                  title="Appliance Energy (Normal vs Anomaly)")
                fig_box1.update_layout(**PLOTLY_THEME, showlegend=False)
                st.plotly_chart(fig_box1, use_container_width=True)
            with col_b:
                # Average Indoor Temp vs Anomaly
                df_anomaly["avg_temp"] = df_anomaly[[c for c in df_anomaly.columns if c.startswith("T") and c[1:].isdigit()]].mean(axis=1)
                fig_box2 = px.box(df_anomaly, x="is_anomaly", y="avg_temp", color="is_anomaly",
                                  color_discrete_map={False: "#6C63FF", True: "#FF6B6B"},
                                  title="Average Indoor Temp (Normal vs Anomaly)")
                fig_box2.update_layout(**PLOTLY_THEME, showlegend=False)
                st.plotly_chart(fig_box2, use_container_width=True)
                
    else:
        st.warning("⚠️ Isolation Forest model not found. Run `python run_pipeline.py` first.")

# ─────────────────────────────────────────────────────────────────────
# PAGE: OPTIMIZATION SIMULATOR (WHAT-IF)
# ─────────────────────────────────────────────────────────────────────
elif "Optimization Simulator" in page:
    st.markdown('<div class="section-header">💡 Prescriptive Analytics: "What-If" Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Simulate the impact of global household changes (e.g., lowering thermostats) over an entire month to estimate total energy and cost savings.</div>', unsafe_allow_html=True)

    models, scaler = load_models()
    if not models:
        st.warning("⚠️ No trained models found. Run `python run_pipeline.py` first.")
    else:
        best_model_name = "Stacking Ensemble" if "Stacking Ensemble" in models else ("LightGBM (Tuned)" if "LightGBM (Tuned)" in models else list(models.keys())[0])
        sim_model = models[best_model_name]
        st.markdown(f"*(Using **{best_model_name}** for simulation)*")
        
        # User input for simulation parameters
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**🌡️ Temperature Adjustment**")
            temp_adj = st.slider("Change all indoor temperatures by (°C):", -5.0, 5.0, -1.0, 0.5)
        with col_s2:
            st.markdown("**💡 Lights Optimization**")
            light_mult = st.slider("Reduce lighting energy by (%):", 0, 100, 20, 5)

        if st.button("🚀 Run 1-Month Simulation", type="primary"):
            with st.spinner("Simulating..."):
                # Take last 30 days of data for simulation
                df_sim_raw = df_eng.tail(30 * 144).copy()
                X_base = df_sim_raw[FEATURE_COLS].copy()
                
                # Base Predictions
                if any(m in best_model_name for m in ["Ridge", "SVR", "MLP", "Linear Regression", "Lasso"]) and scaler is not None:
                    base_preds = sim_model.predict(scaler.transform(X_base))
                else:
                    base_preds = sim_model.predict(X_base)
                base_preds = np.maximum(0, base_preds)
                
                # Apply Modifications
                X_mod = X_base.copy()
                # Adjust temperatures
                temp_features = [c for c in X_mod.columns if c.startswith("T") and c[1:].isdigit()]
                for tc in temp_features:
                    X_mod[tc] = X_mod[tc] + temp_adj
                
                # Adjust lights
                X_mod["lights"] = X_mod["lights"] * (1 - (light_mult/100))
                X_mod["lights_x_hour"] = X_mod["lights"] * X_mod["hour"]
                
                # Adjust differentials and interactions (simplified updates)
                X_mod["avg_indoor_temp"] = X_mod["avg_indoor_temp"] + temp_adj
                for tc in temp_features:
                    if f"dT_{tc}_out" in X_mod.columns: X_mod[f"dT_{tc}_out"] = X_mod[tc] - X_mod["T_out"]
                    if f"dT_{tc}_T1" in X_mod.columns: X_mod[f"dT_{tc}_T1"] = X_mod[tc] - X_mod["T1"]
                
                # Mod Predictions
                if any(m in best_model_name for m in ["Ridge", "SVR", "MLP", "Linear Regression", "Lasso"]) and scaler is not None:
                    mod_preds = sim_model.predict(scaler.transform(X_mod))
                else:
                    mod_preds = sim_model.predict(X_mod)
                mod_preds = np.maximum(0, mod_preds)
                
                # Calculate Savings
                total_base_wh = np.sum(base_preds)
                total_mod_wh = np.sum(mod_preds)
                saved_wh = total_base_wh - total_mod_wh
                saved_kwh = saved_wh / 1000
                savings_pct = (saved_wh / total_base_wh) * 100
                saved_money = saved_kwh * 0.12 # $0.12 / kWh
                
                # Results UI
                st.markdown("### 📊 Simulation Results (30 Days)")
                r_c1, r_c2, r_c3 = st.columns(3)
                with r_c1:
                    st.metric("Total Energy Saved", f"{saved_kwh:.1f} kWh", f"{savings_pct:.1f}%")
                with r_c2:
                    st.metric("Estimated Cost Savings", f"${saved_money:.2f}", "per month")
                with r_c3:
                    carbon = saved_kwh * 0.385 # ~0.385 kg CO2 per kWh
                    st.metric("CO₂ Emissions Prevented", f"{carbon:.1f} kg", "🌱")
                
                # Plot differences
                plot_df = pd.DataFrame({
                    "Date": df_raw["date"].tail(30*144).reset_index(drop=True),
                    "Baseline": base_preds,
                    "Simulated": mod_preds
                })
                # Resample daily for visualization
                plot_df["Day"] = plot_df["Date"].dt.date
                daily_sim = plot_df.groupby("Day")[["Baseline", "Simulated"]].sum().reset_index()
                
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=daily_sim["Day"], y=daily_sim["Baseline"], mode="lines", name="Baseline (Original)", line=dict(color="#FF6B6B")))
                fig_sim.add_trace(go.Scatter(x=daily_sim["Day"], y=daily_sim["Simulated"], mode="lines", name="Simulated (Optimized)", line=dict(color="#00C9A7"), fill="tonexty"))
                fig_sim.update_layout(title="Daily Energy Consumption: Baseline vs Simulated", yaxis_title="Energy (Wh/day)", **PLOTLY_THEME)
                st.plotly_chart(fig_sim, use_container_width=True)
elif "Time-Series" in page:
    st.markdown('<div class="section-header">📈 Time-Series Deep Dive</div>', unsafe_allow_html=True)

    # ── Weekly Pattern ─────────────────────────────────────────────
    df_ts = df_raw.copy()
    df_ts["hour"]  = df_ts["date"].dt.hour
    df_ts["month"] = df_ts["date"].dt.month
    df_ts["week"]  = df_ts["date"].dt.isocalendar().week.astype(int)
    df_ts["date_only"] = df_ts["date"].dt.date

    daily_agg = df_ts.groupby("date_only")["Appliances"].agg(["mean", "max", "min", "sum"]).reset_index()
    daily_agg.columns = ["date", "mean", "max", "min", "total"]

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["max"], name="Daily Max",
                                    line=dict(color="#FF6B6B", width=1), fill=None))
    fig_daily.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["mean"], name="Daily Mean",
                                    line=dict(color="#6C63FF", width=2),
                                    fill="tonexty", fillcolor="rgba(108,99,255,0.1)"))
    fig_daily.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["min"], name="Daily Min",
                                    line=dict(color="#00C9A7", width=1),
                                    fill="tonexty", fillcolor="rgba(0,201,167,0.1)"))
    fig_daily.update_layout(title="Daily Appliance Energy — Min/Mean/Max Bands", **PLOTLY_THEME,
                             legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_daily, use_container_width=True)

    # ── Weekly Total ───────────────────────────────────────────────
    weekly_total = df_ts.groupby("week")["Appliances"].sum().reset_index()
    fig_weekly = px.bar(weekly_total, x="week", y="Appliances",
                         color="Appliances", color_continuous_scale="Plasma",
                         title="Weekly Total Appliance Energy (Wh)",
                         labels={"week": "Week of Year", "Appliances": "Total Energy (Wh)"})
    fig_weekly.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_weekly, use_container_width=True)

    # ── Seasonal Decomposition proxy ──────────────────────────────
    st.markdown('<div class="section-header">📉 Rolling Average Analysis</div>', unsafe_allow_html=True)
    df_ts_sorted = df_ts.sort_values("date")
    roll_win = st.slider("Rolling window (10-min intervals):", 6, 288, 72, step=6,
                          help="72 = 12 hours, 144 = 1 day, 288 = 2 days")

    df_ts_sorted["rolling_avg"] = df_ts_sorted["Appliances"].rolling(roll_win, center=True).mean()
    df_ts_sorted["rolling_std"] = df_ts_sorted["Appliances"].rolling(roll_win, center=True).std()
    sample = df_ts_sorted.tail(2016)  # Last ~2 weeks

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=sample["date"], y=sample["Appliances"], name="Raw",
                                   line=dict(color="#6C63FF", width=0.8), opacity=0.5))
    fig_roll.add_trace(go.Scatter(x=sample["date"], y=sample["rolling_avg"], name="Rolling Mean",
                                   line=dict(color="#F7931E", width=2.5)))
    fig_roll.add_trace(go.Scatter(
        x=pd.concat([sample["date"], sample["date"].iloc[::-1]]),
        y=pd.concat([sample["rolling_avg"] + sample["rolling_std"],
                      (sample["rolling_avg"] - sample["rolling_std"]).iloc[::-1]]),
        fill="toself", fillcolor="rgba(247,147,30,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="±1 Std Dev"
    ))
    fig_roll.update_layout(title=f"Rolling Average (window={roll_win} × 10-min)", **PLOTLY_THEME,
                            legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_roll, use_container_width=True)

    # ── Autocorrelation ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔄 Autocorrelation Analysis</div>', unsafe_allow_html=True)
    n_lags = st.slider("Number of lags to display:", 10, 200, 50)
    acf_vals = pd.Series(df_ts["Appliances"].values).autocorr
    acf_list = [df_ts["Appliances"].autocorr(lag=i) for i in range(1, n_lags + 1)]

    fig_acf = go.Figure()
    for i, v in enumerate(acf_list, 1):
        color = "#6C63FF" if v >= 0 else "#FF6B6B"
        fig_acf.add_trace(go.Scatter(x=[i, i], y=[0, v], mode="lines",
                                      line=dict(color=color, width=2), showlegend=False))
    fig_acf.add_hline(y=0, line_color="white", line_width=0.8)
    ci = 1.96 / np.sqrt(len(df_ts))
    fig_acf.add_hline(y=ci,  line_dash="dash", line_color="#F7931E", line_width=1)
    fig_acf.add_hline(y=-ci, line_dash="dash", line_color="#F7931E", line_width=1,
                       annotation_text="95% CI")
    fig_acf.update_layout(title="Autocorrelation Function (ACF)",
                           xaxis_title="Lag", yaxis_title="Correlation",
                           **PLOTLY_THEME, height=350)
    st.plotly_chart(fig_acf, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE: MODEL EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────
elif "Explainability" in page:
    st.markdown('<div class="section-header">🧠 Model Explainability & Interpretability</div>', unsafe_allow_html=True)

    # Show saved SHAP plots if available
    shap_summary_path = "plots/09_shap_summary.png"
    shap_bar_path     = "plots/10_shap_importance.png"
    fi_path           = "plots/08_feature_importance.png"

    if os.path.exists(shap_summary_path):
        st.markdown("#### 🔵 SHAP Beeswarm Summary Plot")
        st.markdown('<div class="info-box">Each dot represents one prediction sample. Color indicates the feature value (red=high, blue=low). X-position shows the SHAP impact on the prediction.</div>', unsafe_allow_html=True)
        st.image(shap_summary_path, use_column_width=True)
    else:
        st.info("💡 SHAP plots will appear here after running `python run_pipeline.py`")

    if os.path.exists(shap_bar_path):
        st.markdown("#### 📊 SHAP Global Feature Importance")
        st.image(shap_bar_path, use_column_width=True)

    if os.path.exists(fi_path):
        st.markdown("#### 🎯 Tree-based Feature Importance")
        st.image(fi_path, use_column_width=True)

    # ── Feature Importance Table ──────────────────────────────────
    fi = load_feature_importance()
    if fi is not None:
        st.markdown("#### 📋 Feature Importance Values")
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df["Category"] = fi_df["Feature"].apply(lambda x:
            "Lag/Rolling" if any(k in x for k in ["lag", "rolling"]) else
            "Temperature" if x.startswith("T") or x.startswith("dT") else
            "Humidity" if x.startswith("RH") or x.startswith("dRH") else
            "Temporal" if x in ["hour","day_of_week","month","week","day_of_year","is_weekend","quarter","time_of_day"] else
            "Weather" if x in ["To","Pressure","RH_out","Wind speed","Visibility","Tdewpoint"] else
            "Interaction"
        )
        fig_fi_cat = px.treemap(fi_df.head(40), path=["Category", "Feature"],
                                 values="Importance", color="Importance",
                                 color_continuous_scale="Viridis",
                                 title="Feature Importance Treemap by Category")
        fig_fi_cat.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_fi_cat, use_container_width=True)

    # ── SHAP Explanation ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Understanding SHAP Values")
    st.markdown("""
    <div class="info-box">
    <b>What are SHAP values?</b><br>
    SHAP (SHapley Additive exPlanations) assigns each feature a contribution score for each prediction.
    Positive SHAP → feature pushes prediction <b>higher</b>.
    Negative SHAP → feature pushes prediction <b>lower</b>.
    <br><br>
    <b>Key Findings:</b><br>
    🔹 <b>Lag features</b> (recent energy consumption) are the strongest predictors<br>
    🔹 <b>Hour of day</b> captures daily usage cycles<br>
    🔹 <b>Lights</b> positively correlates with appliance usage<br>
    🔹 <b>Outdoor temperature</b> drives HVAC-related consumption<br>
    🔹 <b>Weekend flag</b> shows behavioral consumption differences
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#6060A0; font-size:0.8rem; padding:1rem 0;'>
⚡ Appliance Energy Consumption Predictor &nbsp;|&nbsp; 
Advanced ML Dashboard &nbsp;|&nbsp; 
Powered by LightGBM · XGBoost · SHAP &nbsp;|&nbsp;
<span style='color:#6C63FF'>Built for Energy Intelligence</span>
</div>
""", unsafe_allow_html=True)
