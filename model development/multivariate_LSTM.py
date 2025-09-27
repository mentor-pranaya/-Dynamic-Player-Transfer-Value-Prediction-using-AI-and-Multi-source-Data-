import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. Load Dataset ---
data_dir = r"C:\Users\Abhinav\Desktop\Project\data"
file_name = "final_top_10_player_data.csv"
file_path = os.path.join(data_dir, file_name)

df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

print("\n--- Cleaning Injury Data ---")
df["total_days_injured"] = (
    df["total_days_injured"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .fillna(0)
    .astype(int)
)

scaler_injury = MinMaxScaler()
df[["days_norm", "count_norm"]] = scaler_injury.fit_transform(
    df[["total_days_injured", "injury_count"]]
)
df["injury_index"] = df["days_norm"] + df["count_norm"]
df.drop(columns=["days_norm", "count_norm"], inplace=True)
print("🔧 Injury risk index created!")

df_encoded = pd.get_dummies(df, columns=["position"], prefix="pos")

features = [
    "goals", "assists", "successful_passes", "tackles_won",
    "avg_sentiment_score", "injury_index"
]
features += [c for c in df_encoded.columns if c.startswith("pos_")]

target = "Market Value 2015 (in millions €)"

dataset = df_encoded[features + [target]].copy()
print("📊 Features and target prepared!")

scaler_all = MinMaxScaler()
scaled_data = scaler_all.fit_transform(dataset)

X = scaled_data[:, :-1]
y = scaled_data[:, -1]

X = X.reshape((X.shape[0], 1, X.shape[1]))
print(f"✅ Reshaped X for LSTM: {X.shape}")

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

print("\n🚀 Training LSTM Model...")
history = model.fit(X, y, epochs=120, batch_size=2, verbose=0)
print("✅ Training finished!")

scaled_preds = model.predict(X)

dummy = np.zeros((len(scaled_preds), scaled_data.shape[1]))
dummy[:, -1] = scaled_preds.ravel()
preds = scaler_all.inverse_transform(dummy)[:, -1]

results = pd.DataFrame({
    "Player": df["player_name"],
    "Actual Value (€M)": df[target],
    "Predicted Value (€M)": preds.round(2)
})

print("\n📌 Final Results Preview:")
print(results.head())

out_file = os.path.join(data_dir, "lstm_predictions.csv")
results.to_csv(out_file, index=False)
print(f"\n💾 Predictions saved at: {out_file}")
