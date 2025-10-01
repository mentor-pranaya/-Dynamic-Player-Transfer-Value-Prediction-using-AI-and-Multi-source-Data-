import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- CONFIGURATION ---
TARGET_COL = 'current_value'
# Note: Added position_encoded and winger based on your CSV snippet
FEATURE_COLS = ['age', 'appearance', 'goals', 'assists', 'yellow cards', 'red cards',
                  'minutes played', 'days_injured', 'games_injured',
                  'highest_value', 'position_encoded', 'winger']
TEST_SIZE = 0.2

# --- 1. Data Loading and Robust Preprocessing (KeyError Fix) ---

# Load the dataset
df = pd.read_csv(r"D:\Pythonproject\datasets\pythonProject\final_data.csv")

# *** FIX FOR KEYERROR: Clean column names by stripping whitespace ***
df.columns = df.columns.str.strip()

# Ensure the target column is now present after cleaning
if TARGET_COL not in df.columns:
    print(f"Error: Target column '{TARGET_COL}' not found even after cleaning.")
    print("Please check the exact spelling of 'current_value' in your CSV file.")
    exit()

df_model = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()

# --- 2. Log Transformation and Splitting ---

# CRITICAL FIX: LOG TRANSFORM the Target Variable
df_model['target_log'] = np.log1p(df_model[TARGET_COL])
LOG_TARGET_COL = 'target_log'

# Scale the feature data (X)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_model[FEATURE_COLS])
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
y = df_model[LOG_TARGET_COL]

# Standard Non-Sequential Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=TEST_SIZE, random_state=42
)

# --- 3. Model Training (Random Forest Regressor) ---
print("\n--- Training Random Forest Regressor ---")

# Initialize and train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)

# --- 4. Prediction and Inverse Log Transformation (DELIVERABLE 2) ---

# Generate Log Predictions
y_pred_log = rf_model.predict(X_test)

# Inverse Transform the Log predictions to get real currency values
y_pred_unscaled = np.expm1(y_pred_log)
y_test_unscaled = np.expm1(y_test)

# --- 5. Evaluation (Performance Metrics) (DELIVERABLE 3 - Part 1) ---

# Calculate Metrics
rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
r2 = r2_score(y_test_unscaled, y_pred_unscaled)

print(f"\n--- Random Forest Performance on Test Set ---")
print(f"RMSE (Root Mean Squared Error): €{rmse:,.0f}")
print(f"MAE (Mean Absolute Error):      €{mae:,.0f}")
print(f"R-squared (R2 Score):           {r2:.4f}")

# Create and Display Prediction Results Table
results_df_rf = pd.DataFrame({
    'Actual_Value': y_test_unscaled.round(0),
    'RF_Predicted_Value': y_pred_unscaled.round(0)
})

def format_currency(value):
    return f'€{value:,.0f}'

results_df_display = results_df_rf.head(10).copy()
results_df_display['Actual_Value'] = results_df_display['Actual_Value'].apply(format_currency)
results_df_display['RF_Predicted_Value'] = results_df_display['RF_Predicted_Value'].apply(format_currency)

print("\n--- Initial Prediction Results (First 10 Test Samples) ---")
print("Model: Random Forest Regressor (Non-Sequential)")
print(results_df_display.to_markdown(index=False))

# --- 6. Final Deliverables: Saving Files ---

# Deliverable 1: Save the trained model
model_path = 'final_random_forest_regressor.pkl'
joblib.dump(rf_model, model_path)
print(f"\n✅ Deliverable 1: Trained Model saved to: {model_path}")

# Deliverable 2: Prediction results saved to CSV
results_file_path = 'final_rf_prediction_results.csv'
results_df_rf.to_csv(results_file_path, index=False)
print(f"✅ Deliverable 2: Full prediction results saved to: {results_file_path}")

# Deliverable 3: Actual vs. Predicted Plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(y_test_unscaled, y_pred_unscaled, alpha=0.6, color='forestgreen')
max_val = max(y_test_unscaled.max(), y_pred_unscaled.max()) * 1.05
plot_range = [0, max_val]
plt.plot(plot_range, plot_range, color='red', linestyle='--', label='Ideal Prediction')
plt.title('Random Forest: Actual vs. Predicted Player Value')
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')
plt.ticklabel_format(style='plain', axis='both')
plt.legend()
plt.savefig('rf_actual_vs_predicted_plot.png')
plt.close()

print("✅ Deliverable 3: Actual vs. Predicted plot saved as rf_actual_vs_predicted_plot.png")
