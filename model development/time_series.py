import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

data_path = r"C:/Users/Abhinav/TransferIQ/data/market_value_data.csv"
df = pd.read_csv(data_path, encoding="latin1")

print("🔎 Loaded dataset columns:", list(df.columns))

df["transfer_value"] = pd.to_numeric(df["market_value"], errors="coerce")
numeric_df = df.select_dtypes(include=[np.number]).dropna()

def build_sequences(dataframe, window, target="transfer_value"):
    inputs, targets = [], []
    for start in range(len(dataframe) - window):
        subset = dataframe.iloc[start:start+window].drop(columns=[target], errors="ignore").values
        label = dataframe.iloc[start+window][target]
        inputs.append(subset)
        targets.append(label)
    return np.array(inputs), np.array(targets)

time_window = 12
X, y = build_sequences(numeric_df, time_window)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, shuffle=False)

model = Sequential()
model.add(LSTM(80, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(40))
model.add(Dense(20, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mae") 

print("🚀 Training LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

print("✅ Training finished.")
