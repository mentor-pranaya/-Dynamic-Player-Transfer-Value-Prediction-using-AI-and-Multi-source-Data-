import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. Load Your Clean Data ---
data_folder = 'data'
file_path = os.path.join(data_folder, 'final_top_10_player_data.csv')
df = pd.read_csv(file_path)

# --- 2. Select and Prepare a Single Feature ---
# As per the PDF's 'univariate' suggestion, we'll start with just one feature.
# Let's use 'successful_passes' as it's a good continuous metric.
feature_name = 'successful_passes'
target_name = 'Market Value 2015 (in millions €)'

# Create a new DataFrame with only the feature and target
data = df[[feature_name, target_name]].copy()

print("--- Data Loaded and Prepared ---")
print(data.head())

# --- 3. Scale the Data ---
# Initialize the scaler to transform data to a range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the feature and target columns
data_scaled = scaler.fit_transform(data)

print("\n--- Data Scaled (first 5 rows) ---")
print(data_scaled[:5])

# --- 4. Prepare Data for LSTM ---
# Separate the scaled data back into features (X) and target (y)
X = data_scaled[:, 0]  # The first column (successful_passes)
y = data_scaled[:, 1]  # The second column (market_value)

# Reshape X to the required 3D format: [samples, timesteps, features]
X = X.reshape(X.shape[0], 1, 1)

print("\n--- Data Reshaped for LSTM ---")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# --- 5. Build the LSTM Model ---
# Initialize a Sequential model
model = Sequential()
# Add an LSTM layer with 50 units (a common starting point)
# input_shape=(1, 1) corresponds to our [timesteps, features]
model.add(LSTM(50, input_shape=(1, 1)))
# Add a standard output layer with one neuron to predict the single market value
model.add(Dense(1))

# Compile the model with an optimizer and a loss function
model.compile(optimizer='adam', loss='mean_squared_error')

print("\n--- Training the LSTM Model ---")
# Train the model on the data
# epochs=100 means the model will go through the data 100 times to learn
# verbose=0 will keep the output clean
model.fit(X, y, epochs=100, verbose=0)
print("Model training complete!")

# --- 6. Make and Evaluate Predictions ---
# Make predictions on the entire dataset
predictions_scaled = model.predict(X)

# The scaler was fitted on two columns, so we need to create a dummy array
# to inverse transform our single prediction column.
predictions = scaler.inverse_transform(np.concatenate((X.reshape(X.shape[0], 1), predictions_scaled), axis=1))[:,1]

# Create a final DataFrame to compare actual vs. predicted values
results = pd.DataFrame({
    'Player Name': df['player_name'],
    'Actual Market Value (€M)': df[target_name],
    'Predicted Market Value (€M)': predictions
})

print("\n--- Model Predictions ---")
print(results)