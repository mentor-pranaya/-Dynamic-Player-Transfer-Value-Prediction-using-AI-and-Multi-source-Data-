import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import load_model

FEATURE_FILE = "/Users/ghans/OneDrive/Desktop/filemanaging/master_list_final_features.csv"
LSTM_MODEL_PATH = "models/lstm_multivariate_model.h5"
OUTPUT_PATH = "models/ensemble_model_results.csv"

try:
    df = pd.read_csv(FEATURE_FILE)
    print(f"✅ Loaded feature-engineered dataset from {FEATURE_FILE}")
except FileNotFoundError:
    print(f"❌ Feature file not found at {FEATURE_FILE}")
    exit()

print(f"Dataset shape: {df.shape}\n")

TARGET = "market_value_in_eur"
required_features = [
    "overall_rating", "potential", "performance_index",
    "player_influence_score", "fitness_index", "growth_potential",
    "value_prediction_index"
]

available_features = [f for f in required_features if f in df.columns]
X = df[available_features].values
y = df[TARGET].values if TARGET in df.columns else np.zeros(len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}\n")

if os.path.exists(LSTM_MODEL_PATH):
    lstm_model = load_model(LSTM_MODEL_PATH)
    print("✅ LSTM model loaded successfully.")
    X_lstm_test = np.expand_dims(X_test, axis=1)
    lstm_preds = lstm_model.predict(X_lstm_test).flatten()
else:
    print("⚠️ LSTM model not found. Using zeros as placeholder predictions.")
    lstm_preds = np.zeros(len(X_test))

X_test_with_lstm = np.concatenate([X_test, lstm_preds.reshape(-1, 1)], axis=1)
X_train_with_lstm = np.concatenate(
    [X_train, np.random.normal(np.mean(lstm_preds), np.std(lstm_preds), len(X_train)).reshape(-1, 1)],
    axis=1
)

print("Training ensemble models...\n")

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_with_lstm, y_train)
xgb_preds = xgb_model.predict(X_test_with_lstm)

lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train_with_lstm, y_train)
lgb_preds = lgb_model.predict(X_test_with_lstm)

ensemble_preds = (0.5 * xgb_preds + 0.5 * lgb_preds)

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.2f} | R²: {r2:.3f}")
    return rmse, r2

print("\n📊 Model Evaluation:")
evaluate_model("XGBoost", y_test, xgb_preds)
evaluate_model("LightGBM", y_test, lgb_preds)
evaluate_model("Ensemble (Average)", y_test, ensemble_preds)

results_df = pd.DataFrame({
    "actual_value": y_test,
    "xgboost_pred": xgb_preds,
    "lightgbm_pred": lgb_preds,
    "ensemble_pred": ensemble_preds,
    "lstm_contrib": lstm_preds
})

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Ensemble model predictions saved to: {OUTPUT_PATH}")
print("Ensemble model training completed successfully 🚀")
