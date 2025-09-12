# step2_feature_engineering.py

import pandas as pd
import numpy as np
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = "/Users/veerababu/Downloads/master_list_cleaned.csv"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

df = pd.read_csv(file_path)
print(f"Dataset loaded: {df.shape} rows, {df.shape[1]} columns")

# -----------------------------
# 2. Convert numeric-like columns
# -----------------------------
numeric_cols = ['market_value_in_eur', 'highest_market_value_in_eur', 'age', 'total_injuries', 'total_days_missed', 'experience_years']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert 'Unknown' to NaN

# -----------------------------
# 3. Handle missing values
# -----------------------------
# Numeric columns: fill with median
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill with 'Unknown'
categorical_cols = ['preferred_foot', 'position', 'club', 'most_severe_injury']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# -----------------------------
# 4. Feature Engineering
# -----------------------------

# Interaction feature: age * experience_years
if 'age' in df.columns and 'experience_years' in df.columns:
    df['age_experience'] = df['age'] * df['experience_years']

# Contract risk: 1 if contract_years_remaining < 1 else 0
if 'contract_years_remaining' in df.columns:
    df['contract_risk'] = df['contract_years_remaining'].apply(lambda x: 1 if x < 1 else 0)

# One-hot encoding for categorical columns
one_hot_cols = ['preferred_foot', 'position', 'club']
df = pd.get_dummies(df, columns=[col for col in one_hot_cols if col in df.columns], drop_first=True)

# -----------------------------
# 5. Save processed dataset
# -----------------------------
output_folder = "/Users/veerababu/Downloads/cleaned"
os.makedirs(output_folder, exist_ok=True)
output_file = "master_list_final_features.csv"
output_path = os.path.join(output_folder, output_file)

df.to_csv(output_path, index=False)
print(f"Processed dataset saved to: {output_path}")
