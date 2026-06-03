"""
  Notebook 3: Model Training & Evaluation
  Run as: python 3_model_training.py
  (or open via Jupyter if you prefer)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, warnings, time, os
warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# ── Load engineered data (assumes run_pipeline.py has been run) ──────
print("Loading data...")
df = pd.read_csv("energydata_complete.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)
df.drop(columns=["rv1", "rv2"], inplace=True, errors="ignore")

df["hour"]        = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["week"]        = df["date"].dt.isocalendar().week.astype(int)
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["quarter"]     = df["date"].dt.quarter
df["time_of_day"] = df["hour"].apply(lambda h: 0 if h<6 else(1 if h<12 else(2 if h<18 else 3)))

for lag in [1, 2, 3, 6, 12, 24, 48, 144]:
    df[f"lag_{lag}"] = df["Appliances"].shift(lag)
for w in [3, 6, 12, 24, 48]:
    df[f"rolling_mean_{w}"] = df["Appliances"].shift(1).rolling(w).mean()
    df[f"rolling_std_{w}"]  = df["Appliances"].shift(1).rolling(w).std()
    df[f"rolling_max_{w}"]  = df["Appliances"].shift(1).rolling(w).max()
    df[f"rolling_min_{w}"]  = df["Appliances"].shift(1).rolling(w).min()

for tc in [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]:
    df[f"dT_{tc}_out"] = df[tc] - df["T_out"]
    df[f"dT_{tc}_T1"]  = df[tc] - df["T1"]
for rc in [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()]:
    df[f"dRH_{rc}_out"] = df[rc] - df["RH_out"]

df["lights_x_hour"]    = df["lights"] * df["hour"]
df["T1_RH1_interact"]  = df["T1"] * df["RH_1"]
df["avg_indoor_temp"]  = df[[c for c in df.columns if c.startswith("T") and c[1:].isdigit() and int(c[1:])<=9]].mean(axis=1)
df["avg_indoor_humid"] = df[[c for c in df.columns if c.startswith("RH_") and c[3:].isdigit() and int(c[3:])<=9]].mean(axis=1)
df.drop(columns=["date"], inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURE_COLS = [c for c in df.columns if c != "Appliances"]
X, y = df[FEATURE_COLS], df["Appliances"]
n = len(df)
n_train = int(0.70*n); n_val = int(0.85*n)
X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_val,   y_val   = X.iloc[n_train:n_val], y.iloc[n_train:n_val]
X_test,  y_test  = X.iloc[n_val:], y.iloc[n_val:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-8)))*100
    return {"Model":name,"MAE":round(mae,2),"RMSE":round(rmse,2),"R2":round(r2,4),"MAPE%":round(mape,2)}

results = []
models_dict = {}

def train_and_eval(name, model, X_tr, y_tr, X_te, scaler_needed=False):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te if not scaler_needed else X_test_sc)
    res  = evaluate(name, y_test, pred)
    results.append(res)
    models_dict[name] = model
    print(f"  {name:30s} R2={res['R2']:.4f} MAE={res['MAE']:.1f} RMSE={res['RMSE']:.1f} [{time.time()-t0:.1f}s]")
    return pred

print("\n=== BASELINE MODELS ===")
train_and_eval("Linear Regression", LinearRegression(), X_train_sc, y_train, X_test_sc, True)
train_and_eval("Ridge",             Ridge(alpha=10),    X_train_sc, y_train, X_test_sc, True)
train_and_eval("Lasso",             Lasso(alpha=0.5, max_iter=5000), X_train_sc, y_train, X_test_sc, True)
train_and_eval("Decision Tree",     DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42), X_train, y_train, X_test)

print("\n=== ENSEMBLE MODELS ===")
train_and_eval("Random Forest", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42), X_train, y_train, X_test)

xgb_m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
    verbosity=0, early_stopping_rounds=30)
xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred_xgb = xgb_m.predict(X_test)
res = evaluate("XGBoost", y_test, pred_xgb); results.append(res); models_dict["XGBoost"] = xgb_m
print(f"  {'XGBoost':30s} R2={res['R2']:.4f}")

lgb_m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, n_jobs=-1, random_state=42, verbosity=-1)
try:
    lgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
except Exception:
    lgb_m.fit(X_train, y_train)
pred_lgb = lgb_m.predict(X_test)
res = evaluate("LightGBM", y_test, pred_lgb); results.append(res); models_dict["LightGBM"] = lgb_m
print(f"  {'LightGBM':30s} R2={res['R2']:.4f}")

sample_idx = np.random.choice(len(X_train_sc), 5000, replace=False)
svr_m = SVR(kernel="rbf", C=100, epsilon=5, gamma="scale")
svr_m.fit(X_train_sc[sample_idx], y_train.iloc[sample_idx])
pred_svr = svr_m.predict(X_test_sc)
res = evaluate("SVR", y_test, pred_svr); results.append(res); models_dict["SVR"] = svr_m
print(f"  {'SVR':30s} R2={res['R2']:.4f}")

mlp_m = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
    solver="adam", max_iter=300, early_stopping=True, random_state=42, batch_size=256)
mlp_m.fit(X_train_sc, y_train)
pred_mlp = mlp_m.predict(X_test_sc)
res = evaluate("MLP Neural Network", y_test, pred_mlp); results.append(res); models_dict["MLP Neural Network"] = mlp_m
print(f"  {'MLP Neural Network':30s} R2={res['R2']:.4f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n=== FINAL RESULTS ===")
print(results_df.to_string(index=False))

# Save
for name, model in models_dict.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{fname}.pkl")
joblib.dump(scaler, "models/scaler.pkl")
results_df.rename(columns={"R2": "R²", "MAPE%": "MAPE(%)"}).to_csv("results/model_comparison.csv", index=False)
print("\nAll models saved to models/")
print("Results saved to results/model_comparison.csv")
