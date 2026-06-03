"""
======================================================================
  Appliance Energy Consumption Prediction — Full ML Pipeline
  run_pipeline.py
  
  Run this script to execute the complete machine learning pipeline:
    1. Data Loading & EDA
    2. Feature Engineering
    3. Model Training (9 models)
    4. Hyperparameter Tuning (Optuna)
    5. SHAP Explainability
    6. Saves models, plots, and results
======================================================================
"""

import sys
import io
# Force UTF-8 output on Windows to avoid codec errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, warnings, time, joblib
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# ── Sklearn ──────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ── Boosting ─────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[WARN] xgboost not installed — skipping XGBoost")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("[WARN] lightgbm not installed — skipping LightGBM")

try:
    from catboost import CatBoostRegressor
    CAT_OK = True
except ImportError:
    CAT_OK = False
    print("[WARN] catboost not installed — skipping CatBoost")

# ── SHAP ─────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("[WARN] shap not installed — skipping SHAP analysis")

# ── Optuna ───────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("[WARN] optuna not installed — skipping hyperparameter tuning")

# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#6C63FF",
    "secondary": "#F7931E",
    "accent":    "#00C9A7",
    "danger":    "#FF6B6B",
    "bg":        "#0F0F1A",
    "card":      "#1A1A2E",
    "text":      "#E0E0FF",
}
COLORS = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
          PALETTE["danger"], "#A855F7", "#22D3EE", "#F43F5E", "#84CC16", "#FB923C"]

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["card"],
    "axes.edgecolor":   "#2D2D4E",
    "axes.labelcolor":  PALETTE["text"],
    "text.color":       PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
    "grid.color":       "#2D2D4E",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   11,
})

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 1 | DATA LOADING & OVERVIEW")
print("="*65)

DATA_PATH = "energydata_complete.csv"
df_raw = pd.read_csv(DATA_PATH)
df_raw["date"] = pd.to_datetime(df_raw["date"], dayfirst=True)
df_raw = df_raw.sort_values("date").reset_index(drop=True)

print(f"  Shape        : {df_raw.shape}")
print(f"  Date range   : {df_raw['date'].min()} to {df_raw['date'].max()}")
print(f"  Missing vals : {df_raw.isnull().sum().sum()}")
print(f"  Target stats :")
print(df_raw["Appliances"].describe().to_string())

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — EDA VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 2 | EXPLORATORY DATA ANALYSIS")
print("="*65)

# 2a — Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Appliance Energy Consumption Distribution", fontsize=16, fontweight="bold", color=PALETTE["text"])

axes[0].hist(df_raw["Appliances"], bins=60, color=PALETTE["primary"], edgecolor="none", alpha=0.85)
axes[0].set_xlabel("Energy (Wh)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram of Appliance Energy")

axes[1].hist(np.log1p(df_raw["Appliances"]), bins=60, color=PALETTE["accent"], edgecolor="none", alpha=0.85)
axes[1].set_xlabel("log(Energy + 1)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Log-Transformed Distribution")

plt.tight_layout()
plt.savefig("plots/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/01_target_distribution.png")

# 2b — Time-Series Overview (first 2 weeks)
fig, ax = plt.subplots(figsize=(16, 5))
sample = df_raw.head(2016)
ax.plot(sample["date"], sample["Appliances"], color=PALETTE["primary"], linewidth=0.8, alpha=0.9)
ax.fill_between(sample["date"], sample["Appliances"], alpha=0.2, color=PALETTE["primary"])
ax.set_title("Appliance Energy Consumption — First 2 Weeks", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Energy (Wh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("plots/02_timeseries_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/02_timeseries_overview.png")

# 2c — Correlation Heatmap
fig, ax = plt.subplots(figsize=(18, 14))
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df_raw[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 10, s=80, l=40, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, annot=False,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8},
            square=True, fmt=".2f")
ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/03_correlation_heatmap.png")

# 2d — Hourly & Day-of-week patterns
df_raw["hour"] = df_raw["date"].dt.hour
df_raw["day_of_week"] = df_raw["date"].dt.dayofweek

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
hourly_mean = df_raw.groupby("hour")["Appliances"].mean()
axes[0].bar(hourly_mean.index, hourly_mean.values, color=PALETTE["secondary"], edgecolor="none", alpha=0.85)
axes[0].set_title("Average Energy by Hour of Day", fontweight="bold")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Avg Energy (Wh)")
axes[0].set_xticks(range(0, 24, 2))

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_mean = df_raw.groupby("day_of_week")["Appliances"].mean()
bar_colors = [PALETTE["danger"] if i >= 5 else PALETTE["accent"] for i in range(7)]
axes[1].bar(range(7), dow_mean.values, color=bar_colors, edgecolor="none", alpha=0.85)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)
axes[1].set_title("Average Energy by Day of Week", fontweight="bold")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Avg Energy (Wh)")

plt.tight_layout()
plt.savefig("plots/04_temporal_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/04_temporal_patterns.png")

# 2e — Top Feature Correlations with Target
target_corr = df_raw[numeric_cols].corr()["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = [PALETTE["accent"] if v > 0 else PALETTE["danger"] for v in target_corr.values]
bars = ax.barh(range(len(target_corr)), target_corr.values, color=colors_bar, alpha=0.85)
ax.set_yticks(range(len(target_corr)))
ax.set_yticklabels(target_corr.index, fontsize=10)
ax.set_xlabel("Pearson Correlation with Appliances")
ax.set_title("Top 15 Feature Correlations with Target", fontweight="bold")
ax.axvline(0, color=PALETTE["text"], linewidth=0.8)
plt.tight_layout()
plt.savefig("plots/05_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/05_feature_correlations.png")

# ─────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 3 | FEATURE ENGINEERING")
print("="*65)

df = df_raw.copy()

# Drop random noise variables
df.drop(columns=["rv1", "rv2"], inplace=True, errors="ignore")

# ── Temporal Features ─────────────────────────────────────────────
df["hour"]        = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["week"]        = df["date"].dt.isocalendar().week.astype(int)
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["quarter"]     = df["date"].dt.quarter

# Time-of-day buckets: Night(0-5), Morning(6-11), Afternoon(12-17), Evening(18-23)
def time_bucket(h):
    if h < 6:   return 0
    if h < 12:  return 1
    if h < 18:  return 2
    return 3
df["time_of_day"] = df["hour"].apply(time_bucket)

# ── Lag Features (10-min increments: 1=10min, 6=1h, 12=2h, 144=24h) ──
for lag in [1, 2, 3, 6, 12, 24, 48, 144]:
    df[f"lag_{lag}"] = df["Appliances"].shift(lag)

# ── Rolling Statistics ──────────────────────────────────────────────
for window in [3, 6, 12, 24, 48]:
    df[f"rolling_mean_{window}"] = df["Appliances"].shift(1).rolling(window).mean()
    df[f"rolling_std_{window}"]  = df["Appliances"].shift(1).rolling(window).std()
    df[f"rolling_max_{window}"]  = df["Appliances"].shift(1).rolling(window).max()
    df[f"rolling_min_{window}"]  = df["Appliances"].shift(1).rolling(window).min()

# ── Temperature Differentials ────────────────────────────────────────
temp_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
for tc in temp_cols:
    df[f"dT_{tc}_out"] = df[tc] - df["T_out"]          # vs outdoor
    df[f"dT_{tc}_T1"]  = df[tc] - df["T1"]           # vs kitchen

# ── Humidity Differentials ───────────────────────────────────────────
rh_cols = [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()]
for rc in rh_cols:
    df[f"dRH_{rc}_out"] = df[rc] - df["RH_out"]

# ── Interaction Features ────────────────────────────────────────────
df["lights_x_hour"]    = df["lights"] * df["hour"]
df["T1_RH1_interact"]  = df["T1"] * df["RH_1"]
df["avg_indoor_temp"]  = df[[c for c in df.columns if c.startswith("T") and c[1:].isdigit() and int(c[1:]) <= 9]].mean(axis=1)
df["avg_indoor_humid"] = df[[c for c in df.columns if c.startswith("RH_") and c[3:].isdigit() and int(c[3:]) <= 9]].mean(axis=1)

# Drop date column
df.drop(columns=["date"], inplace=True)

# Drop rows with NaN from lagging
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Final shape after feature engineering: {df.shape}")
print(f"  Total features: {df.shape[1] - 1}")

# ─────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN/TEST SPLIT (Time-aware)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 4 | TRAIN / VALIDATION / TEST SPLIT")
print("="*65)

FEATURE_COLS = [c for c in df.columns if c != "Appliances"]
X = df[FEATURE_COLS]
y = df["Appliances"]

# 70% train, 15% val, 15% test (time-ordered)
n = len(df)
n_train = int(0.70 * n)
n_val   = int(0.85 * n)

X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_val,   y_val   = X.iloc[n_train:n_val], y.iloc[n_train:n_val]
X_test,  y_test  = X.iloc[n_val:], y.iloc[n_val:]

print(f"  Train size      : {len(X_train):,}")
print(f"  Validation size : {len(X_val):,}")
print(f"  Test size       : {len(X_test):,}")

# Scaler (for linear / neural models)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"Model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R²": round(r2, 4), "MAPE(%)": round(mape, 3)}

results = []

# ─────────────────────────────────────────────────────────────────────
# STEP 5 — BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 5 | BASELINE MODELS")
print("="*65)

# 5.1 Linear Regression
print("  Training Linear Regression ...", end=" ")
t0 = time.time()
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
pred_lr = lr.predict(X_test_sc)
res = evaluate("Linear Regression", y_test, pred_lr)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(lr, "models/linear_regression.pkl")

# 5.2 Ridge
print("  Training Ridge Regression ...", end=" ")
t0 = time.time()
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_sc, y_train)
pred_ridge = ridge.predict(X_test_sc)
res = evaluate("Ridge Regression", y_test, pred_ridge)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(ridge, "models/ridge.pkl")

# 5.3 Lasso
print("  Training Lasso Regression ...", end=" ")
t0 = time.time()
lasso = Lasso(alpha=0.5, max_iter=5000)
lasso.fit(X_train_sc, y_train)
pred_lasso = lasso.predict(X_test_sc)
res = evaluate("Lasso Regression", y_test, pred_lasso)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(lasso, "models/lasso.pkl")

# 5.4 Decision Tree
print("  Training Decision Tree ...", end=" ")
t0 = time.time()
dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
res = evaluate("Decision Tree", y_test, pred_dt)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(dt, "models/decision_tree.pkl")

# ─────────────────────────────────────────────────────────────────────
# STEP 6 — ENSEMBLE MODELS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 6 | ENSEMBLE MODELS")
print("="*65)

# 6.1 Random Forest
print("  Training Random Forest ...", end=" ")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2,
                           n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
res = evaluate("Random Forest", y_test, pred_rf)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(rf, "models/random_forest.pkl")

# 6.2 XGBoost
if XGB_OK:
    print("  Training XGBoost ...", end=" ")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, n_jobs=-1, random_state=42, verbosity=0,
        early_stopping_rounds=30
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_xgb = xgb_model.predict(X_test)
    res = evaluate("XGBoost", y_test, pred_xgb)
    results.append(res)
    print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
    joblib.dump(xgb_model, "models/xgboost.pkl")

# 6.3 LightGBM
if LGB_OK:
    print("  Training LightGBM ...", end=" ")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, verbosity=-1
    )
    try:
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(period=-1)])
    except Exception:
        lgb_model.fit(X_train, y_train)
    pred_lgb = lgb_model.predict(X_test)
    res = evaluate("LightGBM", y_test, pred_lgb)
    results.append(res)
    print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
    joblib.dump(lgb_model, "models/lightgbm.pkl")

# 6.4 CatBoost
if CAT_OK:
    print("  Training CatBoost ...", end=" ")
    t0 = time.time()
    cat_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=7,
        l2_leaf_reg=3.0, random_seed=42, verbose=0
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
    pred_cat = cat_model.predict(X_test)
    res = evaluate("CatBoost", y_test, pred_cat)
    results.append(res)
    print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
    joblib.dump(cat_model, "models/catboost.pkl")

# 6.5 SVR (on small sample for speed)
print("  Training SVR (sampled) ...", end=" ")
t0 = time.time()
sample_idx = np.random.choice(len(X_train_sc), min(5000, len(X_train_sc)), replace=False)
svr_model = SVR(kernel="rbf", C=100, epsilon=5, gamma="scale")
svr_model.fit(X_train_sc[sample_idx], y_train.iloc[sample_idx])
pred_svr = svr_model.predict(X_test_sc)
res = evaluate("SVR", y_test, pred_svr)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(svr_model, "models/svr.pkl")

# 6.6 MLP Neural Network
print("  Training MLP Neural Network ...", end=" ")
t0 = time.time()
mlp_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), activation="relu",
    solver="adam", learning_rate_init=0.001, max_iter=300,
    early_stopping=True, validation_fraction=0.1,
    random_state=42, batch_size=256
)
mlp_model.fit(X_train_sc, y_train)
pred_mlp = mlp_model.predict(X_test_sc)
res = evaluate("MLP Neural Network", y_test, pred_mlp)
results.append(res)
print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
joblib.dump(mlp_model, "models/mlp.pkl")

# ─────────────────────────────────────────────────────────────────────
# STEP 6.5 — ADVANCED: ANOMALY DETECTION (Isolation Forest)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 6.5 | ADVANCED: ANOMALY DETECTION (Isolation Forest)")
print("="*65)
print("  Training Isolation Forest for anomaly detection ...", end=" ")
t0 = time.time()
iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
# Fit on the entire dataset (or just train, but usually done on full to find historical anomalies)
# To avoid data leakage, we fit on train and predict on full later in app, or fit on X_train.
iso_forest.fit(X_train_sc)
# Predict anomalies on Test set (-1 is anomaly, 1 is normal)
pred_anomalies = iso_forest.predict(X_test_sc)
anomaly_count = np.sum(pred_anomalies == -1)
print(f"[{time.time()-t0:.1f}s]  Found {anomaly_count} anomalies in test set.")
joblib.dump(iso_forest, "models/isolation_forest.pkl")

# ─────────────────────────────────────────────────────────────────────
# STEP 6.6 — ADVANCED: STACKING META-ENSEMBLE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 6.6 | ADVANCED: STACKING META-ENSEMBLE")
print("="*65)
if LGB_OK and XGB_OK:
    print("  Training Stacking Regressor (LightGBM + XGBoost + Ridge -> Ridge) ...", end=" ")
    t0 = time.time()
    
    estimators = [
        ('lgb', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0)),
        ('ridge', Ridge(alpha=10.0))
    ]
    
    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )
    
    # StackingRegressor handles its own internal CV, so we fit on train
    stack_model.fit(X_train_sc, y_train)
    pred_stack = stack_model.predict(X_test_sc)
    
    res = evaluate("Stacking Ensemble", y_test, pred_stack)
    results.append(res)
    print(f"R²={res['R²']:.4f}  [{time.time()-t0:.1f}s]")
    joblib.dump(stack_model, "models/stacking_regressor.pkl")
else:
    print("  [WARN] Skipping Stacking (requires both LightGBM and XGBoost).")

# ─────────────────────────────────────────────────────────────────────
# STEP 7 — OPTUNA HYPERPARAMETER TUNING (LightGBM)
# ─────────────────────────────────────────────────────────────────────
if OPTUNA_OK and LGB_OK:
    print("\n" + "="*65)
    print("  STEP 7 | OPTUNA HYPERPARAMETER TUNING (LightGBM)")
    print("="*65)

    def lgb_objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 300, 800),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 31, 127),
            "max_depth":          trial.suggest_int("max_depth", 4, 12),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 50),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_jobs": -1, "random_state": 42, "verbosity": -1
        }
        model = lgb.LGBMRegressor(**params)
        try:
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                  lgb.log_evaluation(period=-1)])
        except Exception:
            model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    print("  Running 50 Optuna trials (LightGBM) ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"n_jobs": -1, "random_state": 42, "verbosity": -1})
    print(f"  Best RMSE (val): {np.sqrt(study.best_value):.3f}")
    print(f"  Best params: {best_params}")

    lgb_tuned = lgb.LGBMRegressor(**best_params)
    try:
        lgb_tuned.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                  lgb.log_evaluation(period=-1)])
    except Exception:
        lgb_tuned.fit(X_train, y_train)
    pred_lgb_tuned = lgb_tuned.predict(X_test)
    res = evaluate("LightGBM (Tuned)", y_test, pred_lgb_tuned)
    results.append(res)
    print(f"  Test  R²={res['R²']:.4f}  MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}")
    joblib.dump(lgb_tuned, "models/lightgbm_tuned.pkl")
    BEST_MODEL   = lgb_tuned
    BEST_PRED    = pred_lgb_tuned
    BEST_NAME    = "LightGBM (Tuned)"
else:
    # pick best available
    if LGB_OK:
        BEST_MODEL = lgb_model; BEST_PRED = pred_lgb; BEST_NAME = "LightGBM"
    elif XGB_OK:
        BEST_MODEL = xgb_model; BEST_PRED = pred_xgb; BEST_NAME = "XGBoost"
    else:
        BEST_MODEL = rf; BEST_PRED = pred_rf; BEST_NAME = "Random Forest"

# ─────────────────────────────────────────────────────────────────────
# STEP 8 — RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 8 | RESULTS SUMMARY")
print("="*65)

results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
results_df.to_csv("results/model_comparison.csv", index=False)
print(results_df.to_string(index=False))

# ── Plot model comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

for ax, metric, color in zip(axes, ["R²", "RMSE", "MAE"],
                              [PALETTE["accent"], PALETTE["danger"], PALETTE["secondary"]]):
    sorted_df = results_df.sort_values(metric, ascending=(metric != "R²"))
    bars = ax.barh(sorted_df["Model"], sorted_df[metric], color=color, alpha=0.85, edgecolor="none")
    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel(metric)
    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(bar.get_width() + 0.005 * bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5)

plt.tight_layout()
plt.savefig("plots/06_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/06_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────
# STEP 9 — PREDICTION ANALYSIS (Best Model)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  STEP 9 | PREDICTION ANALYSIS  [{BEST_NAME}]")
print("="*65)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Prediction Analysis — {BEST_NAME}", fontsize=16, fontweight="bold")

# 9a — Actual vs Predicted scatter
ax = axes[0, 0]
ax.scatter(y_test, BEST_PRED, alpha=0.3, s=4, color=PALETTE["primary"])
mn, mx = min(y_test.min(), BEST_PRED.min()), max(y_test.max(), BEST_PRED.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual (Wh)")
ax.set_ylabel("Predicted (Wh)")
ax.set_title("Actual vs Predicted")
ax.legend()

# 9b — Residuals distribution
ax = axes[0, 1]
residuals = y_test.values - BEST_PRED
ax.hist(residuals, bins=60, color=PALETTE["secondary"], alpha=0.85, edgecolor="none")
ax.axvline(0, color="white", linewidth=1.5, linestyle="--")
ax.set_xlabel("Residuals (Wh)")
ax.set_ylabel("Frequency")
ax.set_title("Residual Distribution")

# 9c — Residuals vs Predicted
ax = axes[1, 0]
ax.scatter(BEST_PRED, residuals, alpha=0.3, s=4, color=PALETTE["accent"])
ax.axhline(0, color="white", linewidth=1.5, linestyle="--")
ax.set_xlabel("Predicted (Wh)")
ax.set_ylabel("Residuals (Wh)")
ax.set_title("Residuals vs Predicted")

# 9d — Time-series: Actual vs Predicted (last 500 test points)
ax = axes[1, 1]
show_n = min(500, len(y_test))
ax.plot(range(show_n), y_test.values[-show_n:], color=PALETTE["primary"], linewidth=0.9, label="Actual", alpha=0.9)
ax.plot(range(show_n), BEST_PRED[-show_n:], color=PALETTE["secondary"], linewidth=0.9, label="Predicted", alpha=0.9)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Energy (Wh)")
ax.set_title("Actual vs Predicted (Last 500 samples)")
ax.legend()

plt.tight_layout()
plt.savefig("plots/07_prediction_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/07_prediction_analysis.png")

# ─────────────────────────────────────────────────────────────────────
# STEP 10 — FEATURE IMPORTANCE (Best Model)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 10 | FEATURE IMPORTANCE")
print("="*65)

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_), index=feature_names)
    return None

fi = get_feature_importance(BEST_MODEL, FEATURE_COLS)
if fi is not None:
    fi_top = fi.sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_fi = [PALETTE["primary"]] * 5 + [PALETTE["accent"]] * 10 + [PALETTE["secondary"]] * 10
    ax.barh(range(len(fi_top)), fi_top.values, color=colors_fi[:len(fi_top)], alpha=0.85, edgecolor="none")
    ax.set_yticks(range(len(fi_top)))
    ax.set_yticklabels(fi_top.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top 25 Feature Importances — {BEST_NAME}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/08_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] plots/08_feature_importance.png")
    fi_top.to_csv("results/feature_importance.csv", header=["importance"])
    print("  [OK] results/feature_importance.csv")

# ─────────────────────────────────────────────────────────────────────
# STEP 11 — SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────
if SHAP_OK:
    print("\n" + "="*65)
    print("  STEP 11 | SHAP EXPLAINABILITY")
    print("="*65)

    # Use a subset for speed
    X_shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)

    try:
        explainer = shap.TreeExplainer(BEST_MODEL)
        shap_values = explainer.shap_values(X_shap_sample)
        # shap_values may be a list (classifier) or array (regressor)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        # SHAP Summary plot (beeswarm)
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor(PALETTE["bg"])
        shap.summary_plot(sv, X_shap_sample, show=False, plot_type="dot",
                          max_display=20)
        plt.title(f"SHAP Summary Plot — {BEST_NAME}", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig("plots/09_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] plots/09_shap_summary.png")

        # SHAP Bar plot
        fig = plt.figure(figsize=(10, 7))
        fig.patch.set_facecolor(PALETTE["bg"])
        shap.summary_plot(sv, X_shap_sample, show=False, plot_type="bar",
                          max_display=20)
        plt.title(f"SHAP Feature Importance — {BEST_NAME}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("plots/10_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] plots/10_shap_importance.png")

        # Save SHAP values
        np.save("results/shap_values.npy", sv)
        print("  [OK] results/shap_values.npy")

    except Exception as e:
        print(f"  [WARN] SHAP analysis failed: {e}")

# ─────────────────────────────────────────────────────────────────────
# STEP 12 — CROSS VALIDATION (TimeSeriesSplit)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 12 | CROSS-VALIDATION (TimeSeriesSplit)")
print("="*65)

tscv = TimeSeriesSplit(n_splits=5)

if LGB_OK:
    cv_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63,
                                  n_jobs=-1, random_state=42, verbosity=-1)
elif XGB_OK:
    cv_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1,
                                  random_state=42, verbosity=0)
else:
    cv_model = rf

cv_r2_scores, cv_mae_scores, cv_rmse_scores = [], [], []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_vl = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[val_idx]
    cv_model.fit(X_tr, y_tr)
    p = cv_model.predict(X_vl)
    cv_r2_scores.append(r2_score(y_vl, p))
    cv_mae_scores.append(mean_absolute_error(y_vl, p))
    cv_rmse_scores.append(np.sqrt(mean_squared_error(y_vl, p)))
    print(f"  Fold {fold+1}: R²={cv_r2_scores[-1]:.4f}  MAE={cv_mae_scores[-1]:.2f}  RMSE={cv_rmse_scores[-1]:.2f}")

print(f"\n  CV Mean R²   : {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
print(f"  CV Mean MAE  : {np.mean(cv_mae_scores):.3f} ± {np.std(cv_mae_scores):.3f}")
print(f"  CV Mean RMSE : {np.mean(cv_rmse_scores):.3f} ± {np.std(cv_rmse_scores):.3f}")

cv_results = pd.DataFrame({
    "Fold": range(1, 6),
    "R²": cv_r2_scores, "MAE": cv_mae_scores, "RMSE": cv_rmse_scores
})
cv_results.to_csv("results/cv_results.csv", index=False)

# Plot CV scores
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cross-Validation Results (TimeSeriesSplit)", fontsize=14, fontweight="bold")
for ax, metric, col in zip(axes, ["R²", "MAE", "RMSE"],
                            [PALETTE["accent"], PALETTE["danger"], PALETTE["secondary"]]):
    vals = cv_results[metric].values
    ax.bar(range(1, 6), vals, color=col, alpha=0.85, edgecolor="none")
    ax.axhline(np.mean(vals), color="white", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(vals):.3f}")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Fold")
    ax.set_title(metric)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plots/11_cv_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] plots/11_cv_results.png")

# ─────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ✅ PIPELINE COMPLETE!")
print("="*65)
print(f"\n  📊 Best Model      : {BEST_NAME}")
best_res = results_df.iloc[0]
print(f"  📈 R² Score        : {best_res['R²']}")
print(f"  📉 MAE             : {best_res['MAE']} Wh")
print(f"  📉 RMSE            : {best_res['RMSE']} Wh")
print(f"\n  📁 Saved models    : models/")
print(f"  📁 Saved plots     : plots/")
print(f"  📁 Saved results   : results/")
print(f"\n  💡 Launch dashboard: streamlit run app.py")
print("="*65)
