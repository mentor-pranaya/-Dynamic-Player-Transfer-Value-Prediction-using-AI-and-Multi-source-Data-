# rf_cleaned_updated.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = "/Users/veerababu/Downloads/master_list_cleaned.csv"
df = pd.read_csv(file_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------
# 2. Prepare features and target
# -----------------------------
# Target variable
y = pd.to_numeric(df['market_value_in_eur'], errors='coerce').fillna(0)

# Drop irrelevant/object columns
drop_cols = ['player', 'player_code', 'country_of_birth', 'city_of_birth', 
             'country_of_citizenship', 'date_of_birth', 'contract_expiration_date', 
             'agent_name', 'image_url', 'url', 'current_club_domestic_competition_id', 
             'current_club_name', 'slug', 'player_id', 'most_severe_injury', 
             'all_injuries_details', 'total_games_missed', 'text', 'sentiment', 'score',
             'market_value_in_eur']  # target dropped here

X = df.drop(columns=drop_cols, errors='ignore')

# Identify categorical columns safely
categorical_cols = [col for col in X.select_dtypes(include=['object']).columns]
categorical_cols = list(set(categorical_cols) | set(['sub_position']))  # include sub_position

# Encode categorical columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Ensure all numeric and fill any remaining NaNs
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# -----------------------------
# 3. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# 4. Train Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# -----------------------------
# 6. Feature Importance
# -----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))

# -----------------------------
# 7. Save the trained Random Forest model in Downloads
# -----------------------------
model_path = '/Users/veerababu/Downloads/random_forest_model.pkl'
joblib.dump(model, model_path)
print(f"Random Forest model saved to: {model_path}")

# -----------------------------
# 8. Save feature importances in Downloads
# -----------------------------
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fi_path = '/Users/veerababu/Downloads/feature_importances.csv'
feature_importances.to_csv(fi_path, index=False)
print(f"Feature importances saved to: {fi_path}")

# -----------------------------
# 9. Save predictions on test set in Downloads
# -----------------------------
predictions = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

pred_path = '/Users/veerababu/Downloads/rf_predictions.csv'
predictions.to_csv(pred_path, index=False)
print(f"Predictions saved to: {pred_path}")
