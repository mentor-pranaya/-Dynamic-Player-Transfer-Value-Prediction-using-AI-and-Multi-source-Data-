import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = r"C:\Users\Abhinav\Desktop\Project\data"
X_FILE = os.path.join(BASE_DIR, "X_scaled_features.csv")
Y_FILE = os.path.join(BASE_DIR, "y_target.csv")

print("[1] Loading model-ready data...")
X = pd.read_csv(X_FILE)
y = pd.read_csv(Y_FILE).squeeze()

print(f"✔ Features loaded: {X.shape}")
print(f"✔ Target loaded: {y.shape}")

print("\n[2] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✔ Training set: {X_train.shape}, Testing set: {X_test.shape}")

print("\n[3] Loading LSTM predictions...")

LSTM_TRAIN_FILE = os.path.join(BASE_DIR, "lstm_train_preds.npy")
LSTM_TEST_FILE = os.path.join(BASE_DIR, "lstm_test_preds.npy")

lstm_train_preds = np.load(LSTM_TRAIN_FILE)
lstm_test_preds = np.load(LSTM_TEST_FILE)

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} Evaluation ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f} (explains {r2*100:.1f}% variance)")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []

print("\n[5] Evaluating LSTM baseline...")
results.append(evaluate_model("LSTM Only", y_test, lstm_test_preds))

print("\n[6] Training XGBoost (no LSTM features)...")

xgb_base = XGBRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=6,
    random_state=42
)

xgb_base.fit(X_train, y_train)
y_pred_xgb = xgb_base.predict(X_test)

results.append(evaluate_model("XGBoost Only", y_test, y_pred_xgb))

joblib.dump(xgb_base, os.path.join(BASE_DIR, "xgb_base.pkl"))

print("\n[7] Training Ensemble (XGBoost + LSTM features)...")

X_train_ens = X_train.copy()
X_test_ens = X_test.copy()
X_train_ens["lstm_feature"] = lstm_train_preds
X_test_ens["lstm_feature"] = lstm_test_preds

xgb_ensemble = XGBRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=6,
    random_state=42
)

xgb_ensemble.fit(X_train_ens, y_train)
y_pred_ensemble = xgb_ensemble.predict(X_test_ens)

results.append(evaluate_model("Ensemble (XGB + LSTM)", y_test, y_pred_ensemble))

joblib.dump(xgb_ensemble, os.path.join(BASE_DIR, "xgb_lstm_ensemble.pkl"))

print("\n[8] Saving evaluation results...")
df_results = pd.DataFrame(results)
report_file = os.path.join(BASE_DIR, "ensemble_evaluation_report.csv")
df_results.to_csv(report_file, index=False)
print(f"✔ Report saved to {report_file}")

print("\n=== Pipeline Complete ===")
