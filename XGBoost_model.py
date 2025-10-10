import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib # <-- Import joblib for saving the model

# --- 1. Data Loading and Initial Cleaning ---
try:
    # Load the datasets
    df_21_22 = pd.read_csv('2021_22_features_with_prior_injuries_and_sentiment.csv')
    df_22_23 = pd.read_csv('2022_23_features_with_prior_injuries_and_sentiment.csv')
    df_23_24 = pd.read_csv('2023_24_features_with_prior_injuries_and_sentiment.csv')
    df_market_values_24_25 = pd.read_csv('market_values_24_25.csv')
    print("All CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Please ensure all required CSV files are in the correct directory.")
    exit()

# For this model, we'll use the most recent season's data (23-24) as the primary feature set.
# This dataset contains player performance for the 23-24 season and their market value at the end of that season.
data = df_23_24.copy()

# Clean the target variable: market_value_eur
# Drop rows with missing or zero market value, as we can't train on them.
data.dropna(subset=['market_value_eur'], inplace=True)
data = data[data['market_value_eur'] > 0]

# --- 2. Feature Selection and Engineering ---
# Select a robust set of features that logically influence player value.
# We will use per-90-minute stats to normalize for playing time.
features = [
    'age',
    'mp', # Matches Played
    'starts',
    'min', # Minutes Played
    'pos', # Position
    'gls_90', # Goals per 90
    'ast_90', # Assists per 90
    'xg_90', # Expected Goals per 90
    'xag_90', # Expected Assists per 90
    'prog_carries', # Progressive Carries
    'prog_passes', # Progressive Passes
    'prior_injury_count',
    'total_days_missed_prior',
    'compound' # Overall sentiment score
]
target = 'market_value_eur'

# Create the final DataFrame for the model
model_df = data[features + [target]].copy()
model_df.dropna(inplace=True)

# Log-transform the target variable to handle its skewed distribution, which helps linear models perform better.
model_df['log_market_value'] = np.log1p(model_df[target])
log_target = 'log_market_value'

print(f"\nData prepared for modeling. Shape: {model_df.shape}")

# --- 3. Model Building and Training ---

# Separate features (X) and target (y)
X = model_df[features]
y = model_df[log_target]

# Identify categorical and numerical features
categorical_features = ['pos']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Create a preprocessing pipeline
# OneHotEncoder handles categorical variables, and the rest of the features ('passthrough') are left as is.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the model - we use XGBoost, a powerful and popular gradient boosting model.
xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Create the full model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb_regressor)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining the XGBoost model...")
# Train the model
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- SAVE THE TRAINED MODEL ---
model_filename = 'player_value_model.joblib'
joblib.dump(model_pipeline, model_filename)
print(f"Model successfully saved to '{model_filename}'")
# -----------------------------

# --- 4. Model Evaluation ---

# Make predictions on the test set
y_pred_log = model_pipeline.predict(X_test)

# Inverse transform the predictions and true values to get them back in Euros
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Calculate evaluation metrics
r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print("\n--- Model Evaluation Metrics ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): €{mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): €{rmse:,.2f}")
print("--------------------------------\n")

# --- 5. Visualizations ---

# a) Actual vs. Predicted Values
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=y_test_actual, y=y_pred, alpha=0.6, ax=ax)
ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], '--', color='red', lw=2)
ax.set_title('Actual vs. Predicted Market Value', fontsize=16, weight='bold')
ax.set_xlabel('Actual Market Value (€)', fontsize=12)
ax.set_ylabel('Predicted Market Value (€)', fontsize=12)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
plt.tight_layout()
plt.show()

# b) Feature Importance
# Get feature names after one-hot encoding
try:
    ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([ohe_feature_names, numerical_features])
    
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', ax=ax)
    ax.set_title('Top 15 Most Important Features', fontsize=16, weight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Could not generate feature importance plot: {e}")

# c) Residuals Plot
residuals = y_test_actual - y_pred
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
ax.axhline(0, color='red', linestyle='--')
ax.set_title('Residuals vs. Predicted Values', fontsize=16, weight='bold')
ax.set_xlabel('Predicted Market Value (€)', fontsize=12)
ax.set_ylabel('Residuals (€)', fontsize=12)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"€{int(x/1e6)}M"))
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, loc: f"€{int(y/1e6)}M"))
plt.tight_layout()
plt.show()


# --- 6. Predicting 24/25 Market Values ---

# We need the 23/24 stats to predict the 24/25 values.
# Let's merge the future market value data with the features from the 23/24 season.
predict_data = df_23_24.copy()
predict_data.dropna(subset=features, inplace=True)

# Select only the players we have future data for
predict_data = predict_data[predict_data['player_id'].isin(df_market_values_24_25['player_id'])]

if not predict_data.empty:
    print("\nPredicting market values for the 24/25 season...")
    X_to_predict = predict_data[features]
    
    # Use the trained pipeline to make predictions
    predictions_log = model_pipeline.predict(X_to_predict)
    predictions_eur = np.expm1(predictions_log)
    
    # Create a results DataFrame
    results_df = predict_data[['player_name']].copy()
    results_df['predicted_market_value_24_25'] = predictions_eur
    
    # Merge with actual 24/25 values for comparison, if available in the provided file
    if 'market_value_eur' in df_market_values_24_25.columns:
        results_df = results_df.merge(
            df_market_values_24_25[['player_id', 'market_value_eur']],
            left_on=predict_data['player_id'],
            right_on='player_id',
            how='left'
        ).rename(columns={'market_value_eur': 'actual_market_value_24_25'})
        results_df.drop(columns='key_0', inplace=True)


    results_df = results_df.sort_values('predicted_market_value_24_25', ascending=False)
    
    print("\n--- Top 20 Predicted Player Market Values for 24/25 ---")
    print(results_df.head(20).to_string(formatters={'predicted_market_value_24_25': '€{:,.0f}'.format, 'actual_market_value_24_25': '€{:,.0f}'.format}))
    print("-------------------------------------------------------\n")
else:
    print("\nCould not find matching players from the 23/24 season to predict 24/25 values.")