import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# --- Load Your Final Dataset ---
data_folder = 'data'
final_data_path = os.path.join(data_folder, 'final_top_10_player_data.csv')
df = pd.read_csv(final_data_path)

print("--- Final dataset loaded successfully ---")
print(df.info())

# --- Clean the 'total_days_injured' column ---
print("\n--- Cleaning 'total_days_injured' column ---")

# 1. Convert the column to string type to ensure string methods work
df['total_days_injured'] = df['total_days_injured'].astype(str)

# 2. Extract only the numbers from the text (e.g., "35 days" -> "35")
# .str.extract('(\d+)') uses a regular expression to find all digits
df['total_days_injured'] = df['total_days_injured'].str.extract('(\d+)', expand=False)

# 3. Fill any rows that might have become empty (if they had no numbers) with 0
df['total_days_injured'] = df['total_days_injured'].fillna(0)

# 4. Convert the cleaned column to a proper integer type
df['total_days_injured'] = df['total_days_injured'].astype(int)

print(f"Cleaned 'total_days_injured' - new data type: {df['total_days_injured'].dtype}")

# --- Step 2: One-Hot Encode Categorical Data ---
print("\n--- Applying one-hot encoding ---")

# Identify the actual text-based columns that need encoding
categorical_cols = ['position', 'Nationality']

# Use pd.get_dummies() to convert these columns into a numerical format
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Data after one-hot encoding:")
print(df_encoded.head())

# --- Separate Features and Target ---
print("\n--- Separating features (X) and target (y) ---")

# Rename the market value column for easier access
df_encoded.rename(columns={'Market Value 2015 (in millions €)': 'market_value'}, inplace=True)

# 'y' is our target variable - the value we want to predict
y = df_encoded['market_value']

# 'X' contains all the input features for the model
# We drop the target variable and the non-numeric player_name column
X = df_encoded.drop(columns=['market_value', 'player_name'])

print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)

# --- Scale Numerical Features ---
print("\n--- Scaling all features ---")
# --- ADD THIS DIAGNOSTIC LINE ---
print("\n--- Checking data types of all feature columns before scaling ---")
print(X.info())

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the feature data and transform it
X_scaled = scaler.fit_transform(X)

# For clarity, convert the scaled data (which is a NumPy array) back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Final scaled features ready for modeling (X_scaled_df):")
print(X_scaled_df.head())

# --- Step 5: Save the Model-Ready Data ---
print("\n--- Saving the final scaled data for modeling ---")

# Define the paths for your final output files
X_output_path = os.path.join(data_folder, 'X_scaled_features.csv')
y_output_path = os.path.join(data_folder, 'y_target.csv')

# Save the scaled features DataFrame
# index=False is important to avoid saving an unnecessary index column
X_scaled_df.to_csv(X_output_path, index=False)

# Save the target Series
# We include a header for clarity when we load it later
y.to_csv(y_output_path, index=False, header=True)

print(f"Features saved to: {X_output_path}")
print(f"Target saved to: {y_output_path}")