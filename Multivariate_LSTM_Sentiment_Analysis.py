import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/Data/Sentiment_Analysis/final_featured_dataset.csv')
df = df.sort_values('date')

# Select features and target
feature_cols = [
    'player_age', 'goals', 'assists', 'minutes_played', 'yellow_cards',
    'red_cards', 'season_days_injured', 'total_days_injured',
    'cumulative_days_injured', 'avg_days_injured_prev_seasons', 'bmi',
    'compound_mean', 'polarity_mean'
]
target_col = 'market_value_in_eur'

# Drop rows with missing values
df = df.dropna(subset=feature_cols + [target_col])

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[feature_cols + [target_col]])

# Create sequences
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_features)):
    X.append(scaled_features[i-sequence_length:i, :-1])  # Multivariate features
    y.append(scaled_features[i, -1])  # Target: market_value_in_eur

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))  # Output is market_value_in_eur prediction

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict
predicted = model.predict(X_test)

# Inverse scaling
predicted_full = np.zeros((predicted.shape[0], len(feature_cols) + 1))
predicted_full[:, -1] = predicted[:, 0]

actual_full = np.zeros((y_test.shape[0], len(feature_cols) + 1))
actual_full[:, -1] = y_test

predicted_inversed = scaler.inverse_transform(predicted_full)[:, -1]
actual_inversed = scaler.inverse_transform(actual_full)[:, -1]

# Plot predictions
plt.plot(actual_inversed, label='Actual Market Value')
plt.plot(predicted_inversed, label='Predicted Market Value')
plt.title('Market Value Prediction')
plt.xlabel('Time Step')
plt.ylabel('Market Value (EUR)')
plt.legend()
plt.show()