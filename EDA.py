# eda.py - Exploratory Data Analysis for master_list_cleaned.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Set file path
# -----------------------------
file_path = "/Users/veerababu/Downloads/master_list_cleaned.csv"

# -----------------------------
# 2. Check if file exists
# -----------------------------
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()  # Stop execution if file not found

# -----------------------------
# 3. Load the dataset
# -----------------------------
df = pd.read_csv(file_path)
print(f"Dataset loaded successfully: {file_path}\n")

# -----------------------------
# 4. Quick overview
# -----------------------------
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# 5. Check missing values
# -----------------------------
print("\nMissing Values per Column:")
print(df.isnull().sum())

# -----------------------------
# 6. Convert numeric-like object columns
# -----------------------------
numeric_cols = ['market_value_in_eur', 'age', 'height_in_cm', 'total_injuries', 'total_days_missed']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert 'Unknown' or invalid to NaN

# Check missing values after conversion
print("\nMissing Values after numeric conversion:")
print(df[numeric_cols].isnull().sum())

# Fill missing numeric values with 0 for visualization
df[numeric_cols] = df[numeric_cols].fillna(0)

# -----------------------------
# 7. Visualizations
# -----------------------------

# Histograms
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

# Boxplots to check for outliers
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

# Correlation heatmap
existing_numeric = [col for col in numeric_cols if col in df.columns]
plt.figure(figsize=(10,8))
corr_matrix = df[existing_numeric].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Categorical features
categorical_cols = ['preferred_foot', 'position', 'club']

for col in categorical_cols:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Countplot of {col}')
        plt.show()

# -----------------------------
# 8. Save processed dataset
# -----------------------------
output_folder = "/Users/veerababu/Downloads/cleaned"
output_file = "master_list_cleaned_eda.csv"

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Full path
output_path = os.path.join(output_folder, output_file)

# Save CSV
df.to_csv(output_path, index=False)
print(f"\nProcessed dataset saved to: {output_path}")
