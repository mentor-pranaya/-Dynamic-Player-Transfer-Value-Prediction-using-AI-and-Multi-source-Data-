import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
import joblib

# --- CONFIGURATION ---
TARGET_COL = 'current_value'
FEATURE_COLS = ['age', 'appearance', 'goals', 'assists', 'yellow cards', 'red cards',
                  'minutes played', 'days_injured', 'games_injured',
                  'highest_value', 'position_encoded', 'winger']
TEST_SIZE = 0.2

# --- 1. Data Loading and Preparation ---

# Load the dataset
df = pd.read_csv(r"D:\Pythonproject\datasets\pythonProject\final_data(punaya_murthy).csv")
df.columns = df.columns.str.strip() # Fix for KeyError

# Prepare data with Log Transformation
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()
df_model['target_log'] = np.log1p(df_model[TARGET_COL])
LOG_TARGET_COL = 'target_log'

# Scaling and Splitting
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_model[FEATURE_COLS])
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
y = df_model[LOG_TARGET_COL]

# Split for training and final testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=TEST_SIZE, random_state=42
)

# --- 2. Hyperparameter Tuning using Grid Search (WITH VERBOSE PROGRESS) ---
print("\n--- Starting Grid Search Hyperparameter Tuning (Optimizing Random Forest) ---")
print("Progress will be tracked below (verbose=3):")

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200],         # Number of trees
    'max_depth': [10, 15, None],        # Max depth of each tree
    'min_samples_split': [5, 10],       # Minimum samples required to split a node
    'min_samples_leaf': [2, 5]          # Minimum samples required at a leaf node
}

# Use R-squared as the scoring metric to maximize performance
scorer = make_scorer(r2_score)

# Initialize GridSearchCV
# verbose=3 provides detailed progress updates for each fold and combination
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=3
)

# Run the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator (the final, tuned model)
final_tuned_model = grid_search.best_estimator_

print("\n✅ Tuning Complete!")
print(f"Optimal Parameters Found: {grid_search.best_params_}")

# --- 3. Final Model Testing and Evaluation ---

# Generate Log Predictions using the final tuned model
y_pred_log = final_tuned_model.predict(X_test)

# Inverse Transform to get real currency values
y_pred_unscaled = np.expm1(y_pred_log)
y_test_unscaled = np.expm1(y_test)

# Calculate Final Metrics
final_rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
final_mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
final_r2 = r2_score(y_test_unscaled, y_pred_unscaled)

print(f"\n--- Final Tuned Random Forest Performance ---")
print(f"RMSE (Root Mean Squared Error): €{final_rmse:,.0f}")
print(f"MAE (Mean Absolute Error):      €{final_mae:,.0f}")
print(f"R-squared (R2 Score):           {final_r2:.4f}")

# --- 4. Final Deliverables ---

# DELIVERABLE 1: Save the final tuned model
final_model_path = 'final_tuned_random_forest_model.pkl'
joblib.dump(final_tuned_model, final_model_path)
print(f"\n✅ Deliverable: Final Tuned Model saved to: {final_model_path}")

# DELIVERABLE 2: Model Evaluation Report Data (Comparison)
# The final report requires a comparison. We'll use the final tuned R2 for the comparison table.
lstm_r2 = -0.0996  # Previously reported Multivariate LSTM R2
rf_baseline_r2 = 0.8980 # Previously reported baseline RF R2

print("\n--- Data for Model Evaluation Report (Comparison) ---")
comparison_data = pd.DataFrame({
    'Model': ['Multivariate LSTM', 'Random Forest (Baseline)', 'Random Forest (Final Tuned)'],
    'R-squared Score': [lstm_r2, rf_baseline_r2, final_r2]
})
print(comparison_data.to_markdown(index=False))