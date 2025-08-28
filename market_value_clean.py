import pandas as pd
import glob
import os

# Path to folder containing CSV files
folder_path = "Desktop/Intern/Ai project/Transfermarkt "

# Get list of all CSV files
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Read and combine CSV files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    
    # ---------- Basic Preprocessing ----------
    # 1. Remove duplicate rows
    temp_df.drop_duplicates(inplace=True)
    
    # 2. Handle missing values (example: fill with mean or drop)
    temp_df.fillna(temp_df.mean(numeric_only=True), inplace=True)
    temp_df.fillna("Unknown", inplace=True)  # for categorical
    
    # 3. Standardize column names
    temp_df.columns = temp_df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # 4. Remove extra spaces from string columns
    for col in temp_df.select_dtypes(include="object").columns:
        temp_df[col] = temp_df[col].str.strip()
    
    # Add to list
    df_list.append(temp_df)

# Merge all cleaned DataFrames
combined_df = pd.concat(df_list, ignore_index=True)

# ---------- Optional Further Preprocessing ----------
# 5. Normalize numerical values (example: Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_cols = combined_df.select_dtypes(include="number").columns
combined_df[num_cols] = scaler.fit_transform(combined_df[num_cols])

# 6. Encode categorical columns (example: One-Hot Encoding)
combined_df = pd.get_dummies(combined_df, drop_first=True)

# Save preprocessed dataset
combined_df.to_csv("preprocessed_data.csv", index=False)

print("✅ Preprocessing completed! Final shape:", combined_df.shape)
