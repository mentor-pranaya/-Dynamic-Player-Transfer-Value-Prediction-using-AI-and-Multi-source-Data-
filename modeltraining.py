import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --- Load Your Model-Ready Data ---
data_folder = 'data'
X_path = os.path.join(data_folder, 'X_scaled_features.csv')
y_path = os.path.join(data_folder, 'y_target.csv')

X = pd.read_csv(X_path)
y = pd.read_csv(y_path)

print("--- Model-ready data loaded successfully ---")
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# --- Split Data for Training and Testing ---
# We use train_test_split to randomly divide our data.
# test_size=0.2 means 20% of the data will be used for testing.
# random_state=42 ensures that the split is the same every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("
--- Data split into training and testing sets ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# --- Build and Train the Model ---
# Initialize the XGBoost Regressor model
# n_estimators is the number of trees the model will build.
# learning_rate controls how much the model learns from its mistakes at each step.
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model using the training data
# The .fit() method is where the learning happens.
model.fit(X_train, y_train)

print("
--- Model training complete! ---")

# --- Make Predictions and Evaluate ---
# Use the trained model to predict the market values for the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("
--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} million €")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} million €")
print(f"R-squared (R²): {r2:.2f}")

print("
--- Explanation of Metrics ---")
print(f"MAE means that, on average, the model's predictions are off by about {mae:.2f} million euros.")
print(f"R² of {r2:.2f} means that our model explains approximately {r2*100:.0f}% of the variance in the player market values.")