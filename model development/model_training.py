import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

print("\n[3] Training XGBoost regressor...")

xgb_model = XGBRegressor(
    n_estimators=150,  
    learning_rate=0.08,
    max_depth=6,   
    random_state=42
)

xgb_model.fit(X_train, y_train)

print("✔ Model training complete")

print("\n[4] Evaluating model...")

y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"MAE  : {mae:.2f} M€ (avg prediction error)")
print(f"RMSE : {rmse:.2f} M€ (penalizes big errors more)")
print(f"R²   : {r2:.2f} (explains {r2*100:.1f}% of variance)")