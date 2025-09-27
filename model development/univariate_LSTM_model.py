import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_DIR = r"C:\Users\Abhinav\Desktop\Project\data"
INPUT_FILE = os.path.join(DATA_DIR, "final_top_10_player_data.csv")

print("[1] Loading dataset...")
df = pd.read_csv(INPUT_FILE)

feature_col = "successful_passes"
target_col = "Market Value 2015 (in millions €)"

if feature_col not in df.columns or target_col not in df.columns:
    raise ValueError(f"Required columns missing in {INPUT_FILE}")

subset = df[[feature_col, target_col]].copy()
print(f"✔ Loaded {len(subset)} rows")

print("\n[2] Scaling features & target...")
scaler = MinMaxScaler()

scaled = scaler.fit_transform(subset)
X_vals = scaled[:, 0]
y_vals = scaled[:, 1]

X_reshaped = X_vals.reshape(-1, 1, 1)

print(f"✔ Reshaped X: {X_reshaped.shape}, y: {y_vals.shape}")

print("\n[3] Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(1, 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

print("\n[4] Training model...")
model.fit(X_reshaped, y_vals, epochs=80, batch_size=4, verbose=0)
print("✔ Training complete")

print("\n[5] Making predictions...")

pred_scaled = model.predict(X_reshaped, verbose=0)

dummy_input = np.hstack((X_vals.reshape(-1, 1), pred_scaled))
predictions = scaler.inverse_transform(dummy_input)[:, 1]

results = pd.DataFrame({
    "player_name": df["player_name"],
    "actual_value": df[target_col],
    "predicted_value": predictions
})

print("\n=== Sample Results ===")
print(results.head())

OUTPUT_FILE = os.path.join(DATA_DIR, "lstm_predictions.csv")
results.to_csv(OUTPUT_FILE, index=False)
print(f"\n✔ Predictions saved to {OUTPUT_FILE}")
