# 🔋 Appliance Energy Consumption Prediction — Advanced ML Project

## 📌 Overview
An end-to-end machine learning system to predict household appliance energy consumption using environmental sensor data, time-series features, and multiple advanced ML algorithms.

## 📂 Dataset
- **Source**: `energydata_complete.csv` (19,735 rows × 29 columns)
- **Target**: `Appliances` (energy in Wh)
- **Sampling**: 10-minute intervals

## 🧪 Features Used
| Category | Features |
|---|---|
| Indoor Temp | T1–T9 (kitchen, living room, laundry, office, bathroom, outside-N, ironing, teen room, parents) |
| Indoor Humidity | RH_1–RH_9 |
| Outdoor Weather | To, Pressure, RH_out, Wind speed, Visibility, Tdewpoint |
| Lighting | lights |
| Temporal | hour, day_of_week, month, is_weekend, time_of_day |
| Lag/Rolling | lag_1, lag_2, rolling_mean_3, rolling_std_6, etc. |

## 🤖 Models Trained
1. Linear Regression (Baseline)
2. Ridge / Lasso Regression
3. Decision Tree
4. **Random Forest** ⭐
5. **XGBoost** ⭐
6. **LightGBM** ⭐ (Best performer)
7. CatBoost
8. SVR
9. MLP Neural Network

## 📊 Key Results
Best model metrics (LightGBM tuned):
- R² Score: ~0.97
- MAE: ~11 Wh
- RMSE: ~22 Wh

## 📁 Project Structure
```
priyanshu/
├── 1_EDA_and_Preprocessing.ipynb
├── 2_Feature_Engineering.ipynb
├── 3_Model_Training_and_Evaluation.ipynb
├── 4_Hyperparameter_Tuning.ipynb
├── 5_Model_Explainability_SHAP.ipynb
├── app.py                    ← Interactive Streamlit Dashboard
├── requirements.txt
├── models/                   ← Saved .pkl model files
└── README.md
```

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Launch interactive dashboard
streamlit run app.py
```

## 🔍 Explainability
- SHAP (SHapley Additive exPlanations) used for global & local feature importance
- Partial Dependence Plots for top features
- Time-series decomposition and residual analysis
