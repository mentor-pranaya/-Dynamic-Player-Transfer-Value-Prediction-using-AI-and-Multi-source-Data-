import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
FILE_PATH = "/Users/veerababu/Downloads/cleaned/master_list_final_features.csv"
TIME_STEPS = 12  # Look-back window (T past steps for all features)
TRAIN_RATIO = 0.8
TARGET_COLUMN = 'market_value_in_eur'
# --- MODEL SAVE PATHS (All set to /Users/veerababu/Downloads) ---
MODEL_SAVE_PATH = '/Users/veerababu/Downloads/multivariate_lstm_model.h5' 
RESULTS_SAVE_PATH = '/Users/veerababu/Downloads/multivariate_lstm_results.csv'
PREDICTIONS_SAVE_PATH = '/Users/veerababu/Downloads/multivariate_lstm_predictions.csv'
# -----------------------

# --- Define Multivariate Features ---
# CRITICAL: Adjust these columns if your final features list is different.
FEATURE_COLUMNS = [
    TARGET_COLUMN, # Must be the first column for simplified indexing later
    'age', 
    'total_injuries', 
    'total_days_missed',
    'contract_years_remaining', 
    'goals', 
    'assists',
    'sentiment_score' # Assuming you have calculated this score
]
FEATURES = len(FEATURE_COLUMNS) 
print(f"Total features being used for Multivariate LSTM: {FEATURES}")


def create_multivariate_sequences(data, time_steps, target_name):
    """Convert multi-feature time-series data into sequences."""
    X, y = [], []
    target_idx = data.columns.get_loc(target_name)
    
    for i in range(len(data) - time_steps):
        # X: Input sequence (T past time steps for ALL features)
        X.append(data.iloc[i:(i + time_steps)].values)
        # y: Output value (The single scaled market value in the next time step)
        y.append(data.iloc[i + time_steps, target_idx])
    return np.array(X), np.array(y)

# =============================================================
# 1. Data Loading and Preparation
# =============================================================
try:
    df = pd.read_csv(FILE_PATH)
    # Ensure data is sorted chronologically by player
    df = df.sort_values(by=['player', 'date'])
    df_features = df[FEATURE_COLUMNS].copy()
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

# Handle NaNs: Fill missing values with the mean before scaling
df_features = df_features.fillna(df_features.mean())

# Scale ALL selected features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df_features.values)
scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLUMNS)

# Create sequences
X, y = create_multivariate_sequences(scaled_df, TIME_STEPS, TARGET_COLUMN)

# Chronological Train/Validation Split
split_index = int(TRAIN_RATIO * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print(f"Total sequences created: {len(X)}")
print(f"Multivariate X_train shape: {X_train.shape}")
print(f"Multivariate X_val shape: {X_val.shape}")

# =============================================================
# 2. Define Multivariate LSTM Model
# =============================================================
model_multi = Sequential()
model_multi.add(LSTM(
    units=100, # Increased units for more complex, multi-feature data
    activation='relu', 
    input_shape=(TIME_STEPS, FEATURES) # Input shape uses N features
))
model_multi.add(Dropout(0.3)) 
model_multi.add(Dense(units=1))

model_multi.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("\nMultivariate Model Summary:")
model_multi.summary()

# =============================================================
# 3. Training and Saving
# =============================================================
# Ensure the directory for the save path exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("\nStarting Multivariate LSTM Training...")
history_multi = model_multi.fit(
    X_train,
    y_train,
    epochs=100, 
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\nMultivariate LSTM Training Complete. Model saved to:", MODEL_SAVE_PATH)

# =============================================================
# 4. Evaluation and Results Saving
# =============================================================

# Predict on validation set
y_pred_scaled = model_multi.predict(X_val).flatten()

# Rescale both true and predicted values back to original EUR scale
target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

# Rescale True Values
y_val_rescaled_full = np.zeros((len(y_val), FEATURES))
y_val_rescaled_full[:, target_idx] = y_val
y_true_eur = scaler.inverse_transform(y_val_rescaled_full)[:, target_idx]

# Rescale Predicted Values
y_pred_rescaled_full = np.zeros((len(y_pred_scaled), FEATURES))
y_pred_rescaled_full[:, target_idx] = y_pred_scaled
y_pred_eur = scaler.inverse_transform(y_pred_rescaled_full)[:, target_idx]


# Calculate Final Metrics (in EUR)
rmse_eur = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)
r2 = r2_score(y_true_eur, y_pred_eur)


print("\n--- Final Multivariate LSTM Results (in EUR) ---")
print(f"RMSE: {rmse_eur:,.2f} EUR")
print(f"MAE: {mae_eur:,.2f} EUR")
print(f"R-squared: {r2:.4f}")

# --- Save Results to CSV (Deliverable Requirement) ---
results_df = pd.DataFrame({
    'Metric': ['RMSE_EUR', 'MAE_EUR', 'R2_Score', 'Final_Val_Loss_Scaled'],
    'Value': [rmse_eur, mae_eur, r2, history_multi.history['val_loss'][-1]]
})
results_df.to_csv(RESULTS_SAVE_PATH, index=False)
print(f"Evaluation metrics saved to: {RESULTS_SAVE_PATH}")

# --- Save Predictions to CSV ---
predictions_df = pd.DataFrame({
    'Actual_EUR': y_true_eur,
    'Predicted_EUR': y_pred_eur
})
predictions_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
print(f"Predictions saved to: {PREDICTIONS_SAVE_PATH}")


# Plotting Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(history_multi.history['loss'], label='Train Loss')
plt.plot(history_multi.history['val_loss'], label='Validation Loss')
plt.title('Multivariate LSTM Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
