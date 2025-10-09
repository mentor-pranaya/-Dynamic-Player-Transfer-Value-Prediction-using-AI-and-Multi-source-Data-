import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# CONFIGURATION
# -----------------------------
FILE_PATH = "/Users/veerababu/Downloads/sample_master_list.csv"
TIME_STEPS = 3  # Reduced to 3 since most players have only 1 row
TRAIN_RATIO = 0.8
TARGET_COLUMN = 'market_value_in_eur'

FEATURE_COLUMNS = [
    TARGET_COLUMN, 
    'age',         
    'total_injuries', 
    'total_days_missed',
    'contract_years_remaining'
]

FEATURES = len(FEATURE_COLUMNS)

# Paths to save model and outputs
MODEL_SAVE_PATH = '/Users/veerababu/Downloads/multivariate_lstm_model.h5'
RESULTS_SAVE_PATH = '/Users/veerababu/Downloads/multivariate_lstm_results.csv'
PREDICTIONS_SAVE_PATH = '/Users/veerababu/Downloads/multivariate_lstm_predictions.csv'

# -----------------------------
# FUNCTIONS
# -----------------------------
def create_sequences(data, time_steps, target_name):
    """Convert multivariate time-series data into sequences."""
    X, y = [], []
    target_idx = data.columns.get_loc(target_name)
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i+time_steps].values)
        y.append(data.iloc[i+time_steps, target_idx])
    return np.array(X), np.array(y)

# -----------------------------
# 1. LOAD AND PREPARE DATA
# -----------------------------
df = pd.read_csv(FILE_PATH)
print(f"Loaded data shape: {df.shape}")

# Ensure all feature columns exist
missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

df_features = df[FEATURE_COLUMNS].copy()

# Convert to numeric & fill NaNs
df_features = df_features.apply(pd.to_numeric, errors='coerce')
df_features = df_features.fillna(df_features.mean(numeric_only=True))

# Scale all features
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(df_features.values)
scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLUMNS)

# Create sequences for LSTM
X, y = create_sequences(scaled_df, TIME_STEPS, TARGET_COLUMN)

if len(X) == 0:
    raise ValueError("No sequences created. Reduce TIME_STEPS or check your data.")

# Train/validation split
split_index = int(TRAIN_RATIO * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"Total sequences created: {len(X)}")
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# -----------------------------
# 2. DEFINE MULTIVARIATE LSTM MODEL
# -----------------------------
model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(TIME_STEPS, FEATURES)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# -----------------------------
# 3. TRAINING
# -----------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"Model training complete. Saved to: {MODEL_SAVE_PATH}")

# -----------------------------
# 4. EVALUATION
# -----------------------------
y_pred_scaled = model.predict(X_val).flatten()

# Rescale to original EUR values
target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

# True values rescaling
y_val_full = np.zeros((len(y_val), FEATURES))
y_val_full[:, target_idx] = y_val
y_true_eur = scaler.inverse_transform(y_val_full)[:, target_idx]

# Predicted values rescaling
y_pred_full = np.zeros((len(y_pred_scaled), FEATURES))
y_pred_full[:, target_idx] = y_pred_scaled
y_pred_eur = scaler.inverse_transform(y_pred_full)[:, target_idx]

# Metrics
rmse_eur = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)
r2 = r2_score(y_true_eur, y_pred_eur)

print("\n--- Multivariate LSTM Results ---")
print(f"RMSE: {rmse_eur:,.2f} EUR")
print(f"MAE: {mae_eur:,.2f} EUR")
print(f"R2 Score: {r2:.4f}")

# Save evaluation metrics
results_df = pd.DataFrame({
    'Metric': ['RMSE_EUR', 'MAE_EUR', 'R2_Score', 'Final_Val_Loss_Scaled'],
    'Value': [rmse_eur, mae_eur, r2, history.history['val_loss'][-1]]
})
results_df.to_csv(RESULTS_SAVE_PATH, index=False)
print(f"Metrics saved to: {RESULTS_SAVE_PATH}")

# Save predictions
pred_df = pd.DataFrame({'Actual_EUR': y_true_eur, 'Predicted_EUR': y_pred_eur})
pred_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
print(f"Predictions saved to: {PREDICTIONS_SAVE_PATH}")

# Plot Loss Curves
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Multivariate LSTM Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
