"""
eda.py
------
Performs Exploratory Data Analysis (EDA) on the merged player dataset.
Analyzes player performance, sentiment, and market value relationships.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Set File Path
# -----------------------------
file_path = r"C:\Users\ghans\OneDrive\Desktop\filemanaging\fifa_players_data_no_duplicates.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv(file_path)
print(f"✅ Dataset loaded successfully! Shape: {df.shape}\n")

# -----------------------------
# 3. Basic Overview
# -----------------------------
print("📋 Column Information:")
print(df.info(), "\n")

print("📊 Summary Statistics:")
print(df.describe().T, "\n")

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
missing = df.isnull().sum()
print("🔍 Missing Values per Column:\n", missing[missing > 0])

# Fill missing numeric values with mean (for EDA only)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# -----------------------------
# 5. Key Numeric Columns for Analysis
# -----------------------------
key_cols = ["age", "overall_rating", "potential", "value_euro", "wage_euro"]

# Check existence of these columns
key_cols = [col for col in key_cols if col in df.columns]
print(f"📈 Numeric columns used in EDA: {key_cols}\n")

# -----------------------------
# 6. Distribution Plots
# -----------------------------
for col in key_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df[key_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap among Key Player Metrics")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Categorical Analysis
# -----------------------------
cat_cols = ["preferred_foot", "nationality", "positions"]

for col in cat_cols:
    if col in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(y=df[col], order=df[col].value_counts().head(10).index)
        plt.title(f"Top 10 {col} Categories")
        plt.tight_layout()
        plt.show()

# -----------------------------
# 9. Market Value vs Rating/Potential
# -----------------------------
if "value_euro" in df.columns and "overall_rating" in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["overall_rating"], y=df["value_euro"])
    plt.title("Player Market Value vs Overall Rating")
    plt.xlabel("Overall Rating")
    plt.ylabel("Market Value (Euro)")
    plt.tight_layout()
    plt.show()

if "potential" in df.columns and "value_euro" in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["potential"], y=df["value_euro"])
    plt.title("Player Market Value vs Potential")
    plt.xlabel("Potential")
    plt.ylabel("Market Value (Euro)")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 10. Save Cleaned Copy
# -----------------------------
output_dir = r"C:\Users\ghans\OneDrive\Desktop\filemanaging\data_cleaning_preprocessing"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "fifa_players_data_eda.csv")

df.to_csv(output_path, index=False)
print(f"✅ Cleaned EDA dataset saved to: {output_path}")
