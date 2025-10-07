"""
lstm_multivariate_model.py
--------------------------
Trains a multivariate LSTM model using player performance,
injury, contract, and sentiment features to predict market value.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================================================================
# 1. Configuration
# ================================================================
BASE_DIR = r"C:\Users\ghans\OneDrive\Desktop\filemanaging"

# Input & Output file paths
FILE_PATH = os.path.join(BASE_DIR, "fifa_players_data_no_duplicates.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "multivariate_lstm_model.h5")
RESULTS_SAVE_PATH = os.path.join(BASE_DIR, "results", "multivariate_lstm_results.csv")
PREDICTIONS_SAVE_PATH = os.path.join(BASE_DIR, "results", "multivariate_lstm_predictions.csv")

# Create folders if missing
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)

# Hyperparameters
TIME_STEPS = 10        # Sequence length
TRAIN_RATIO = 0.8      # Train-validation split ratio
TARGET_COLUMN = "value_euro"

# ================================================================
# 2. Feature Selection (Modify based on your project data)
# ================================================================
FEATURE_COLUMNS = [
    TARGET_COLUMN,       # target feature
    "age",
    "overall_rating",
    "potential",
    "sentiment_impact",  # from feature_engineering
    "total_injuries",
    "total_days_missed",
    "contract_years_remaining"
]
FEATURES = len(FEATURE_COLUMNS)
print(f"✅ Using {FEATURES} features for multivariate LSTM\n")

# ================================================================
# 3. Data Loading & Preparation
# ================================================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)
print(f"📂 Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# Sort by player and date if available
sort_cols = [col for col in ["player", "date"] if col in df.columns]
if sort_cols:
    df = df.sort_values(by=sort_cols)

# Keep only selected features
df_features = df[[col for col in FEATURE_COLUMNS if col in df.columns]].copy()

# Handle missing values
df_features = df_features.fillna(df_features.mean())

# ================================================================
# 4. Scaling
# ================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df_features.values)
scaled_df = pd.DataFrame(scaled_features, columns=df_features.columns)


# ================================================================
# 5. Sequence Creation
# ================================================================
def create_multivariate_sequences(data, time_steps, target_name):
    """Convert feature data into time-step sequences for LSTM."""
    X, y = [], []
    target_idx = data.columns.get_loc(target_name)
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i + time_steps].values)
        y.append(data.iloc[i + time_steps, target_idx])
    return np.array(X), np.array(y)

X, y = create_multivariate_sequences(scaled_df, TIME_STEPS, TARGET_COLUMN)

split_index = int(TRAIN_RATIO * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"🧠 Total sequences: {len(X)}")
print(f"📊 Train shape: {X_train.shape}, Validation shape: {X_val.shape}\n")

# ================================================================
# 6. Model Definition
# ================================================================
model = Sequential([
    LSTM(128, activation='relu', input_shape=(TIME_STEPS, FEATURES)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("✅ Model Summary:")
model.summary()

# ================================================================
# 7. Model Training
# ================================================================
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1)

print("\n🚀 Starting LSTM training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)
print(f"\n✅ Training complete. Best model saved at: {MODEL_SAVE_PATH}")

# ================================================================
# 8. Evaluation
# ================================================================
y_pred_scaled = model.predict(X_val).flatten()
target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

# Reverse scaling
def inverse_scale(y_scaled):
    temp = np.zeros((len(y_scaled), FEATURES))
    temp[:, target_idx] = y_scaled
    return scaler.inverse_transform(temp)[:, target_idx]

y_true = inverse_scale(y_val)
y_pred = inverse_scale(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n📈 Final Evaluation Metrics:")
print(f"RMSE: {rmse:,.2f} EUR")
print(f"MAE: {mae:,.2f} EUR")
print(f"R²: {r2:.4f}")

# ================================================================
# 9. Save Results & Predictions
# ================================================================
results_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2"],
    "Value": [rmse, mae, r2]
})
results_df.to_csv(RESULTS_SAVE_PATH, index=False)

predictions_df = pd.DataFrame({
    "Actual_EUR": y_true,
    "Predicted_EUR": y_pred
})
predictions_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)

print(f"💾 Results saved to: {RESULTS_SAVE_PATH}")
print(f"💾 Predictions saved to: {PREDICTIONS_SAVE_PATH}")

# ================================================================
# 10. Plot Training Curves
# ================================================================
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()
