# data_cleaning_preprocessing.py
# -----------------------------------------------
# Data Cleaning and Preprocessing Script
# For: Dynamic Player Transfer Value Prediction Project
# -----------------------------------------------

import pandas as pd
import numpy as np
import os

# -----------------------------
# 1. Define File Paths
# -----------------------------
RAW_DATA_PATH = "/Users/ghans/OneDrive/Desktop/filemanaging/fifa_players_data.xlsx"
INJURY_DATA_PATH = "/Users/ghans/OneDrive/Desktop/filemanaging/market_injury.py"  # If merged data source available
TRANSFER_DATA_PATH = "/Users/ghans/OneDrive/Desktop/filemanaging/merge_transfer_data.py"

# Output path for cleaned dataset
OUTPUT_FOLDER = "/Users/ghans/OneDrive/Desktop/filemanaging"
OUTPUT_FILE = "master_list_cleaned.csv"

# -----------------------------
# 2. Load Datasets
# -----------------------------
try:
    print("Loading main player dataset...")
    df_players = pd.read_excel(RAW_DATA_PATH)

    # Check if additional data sources exist before reading
    if os.path.exists(TRANSFER_DATA_PATH.replace('.py', '.csv')):
        df_transfer = pd.read_csv(TRANSFER_DATA_PATH.replace('.py', '.csv'))
    else:
        df_transfer = pd.DataFrame()

    print("Datasets loaded successfully.\n")

except FileNotFoundError:
    print(f"Error: One or more files not found. Please check your file paths.")
    exit()

# -----------------------------
# 3. Inspect Data
# -----------------------------
print("Initial Player Data Shape:", df_players.shape)
print("Columns in Dataset:")
print(df_players.columns.tolist())

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
print("\nHandling missing values...")

# Fill numeric columns with mean
numeric_cols = df_players.select_dtypes(include=[np.number]).columns
df_players[numeric_cols] = df_players[numeric_cols].fillna(df_players[numeric_cols].mean())

# Fill object (categorical) columns with mode
for col in df_players.select_dtypes(include=['object']).columns:
    df_players[col] = df_players[col].fillna(df_players[col].mode()[0] if not df_players[col].mode().empty else "Unknown")

print("Missing values handled successfully.\n")

# -----------------------------
# 5. Remove Duplicates
# -----------------------------
before_duplicates = len(df_players)
df_players = df_players.drop_duplicates()
after_duplicates = len(df_players)

print(f"Removed {before_duplicates - after_duplicates} duplicate rows.\n")

# -----------------------------
# 6. Data Type Conversions
# -----------------------------
print("Converting data types where necessary...")

# Convert dates to datetime (if columns exist)
for date_col in ['date', 'contract_expiry', 'birth_date']:
    if date_col in df_players.columns:
        df_players[date_col] = pd.to_datetime(df_players[date_col], errors='coerce')

# Convert market value to numeric (remove symbols if needed)
if 'market_value_in_eur' in df_players.columns:
    df_players['market_value_in_eur'] = (
        df_players['market_value_in_eur']
        .astype(str)
        .str.replace('[^0-9.]', '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )

print("Data types converted successfully.\n")

# -----------------------------
# 7. Feature Adjustments
# -----------------------------
print("Creating additional helper features...")

if {'height_in_cm', 'weight_in_kg'}.issubset(df_players.columns):
    df_players['bmi'] = (df_players['weight_in_kg'] / (df_players['height_in_cm'] / 100) ** 2).round(2)

if 'contract_expiry' in df_players.columns:
    df_players['contract_years_remaining'] = (
        (pd.to_datetime('today') - df_players['contract_expiry']).dt.days / 365.25
    ).abs().round(1)

print("New helper features added.\n")

# -----------------------------
# 8. Merge With Transfer or Injury Data (if available)
# -----------------------------
if not df_transfer.empty:
    print("Merging player data with transfer data...")
    merge_key = 'player' if 'player' in df_transfer.columns else 'name'
    df_merged = pd.merge(df_players, df_transfer, on=merge_key, how='left')
else:
    df_merged = df_players.copy()

print("Merged dataset shape:", df_merged.shape)

# -----------------------------
# 9. Final Dataset Overview
# -----------------------------
print("\nFinal Cleaned Dataset Summary:")
print(df_merged.info())
print(df_merged.describe().T.head())

# -----------------------------
# 10. Save Cleaned Dataset
# -----------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

df_merged.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved successfully at:\n{output_path}")
