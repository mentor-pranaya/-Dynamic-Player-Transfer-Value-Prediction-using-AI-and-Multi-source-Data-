import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv(r'Data\Market value data.csv', encoding='latin1')

# Print actual columns to help you update required_features
print("Columns in your CSV:", df.columns.tolist())

# Update this list after you see the printed columns
required_features = ['market_value']  # Add real feature columns here

assert all(col in df.columns for col in required_features), "Missing required columns!"

# Use only relevant features
df_multivariate = df[required_features]

def create_sequences(df, seq_length, target_col):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df.iloc[i:i+seq_length].drop(target_col, axis=1, errors='ignore').values)
        y.append(df.iloc[i+seq_length][target_col])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(df_multivariate, sequence_length, 'market_value')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Re-train LSTM with multivariate input
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)