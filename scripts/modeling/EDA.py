import pandas as pd

# ------------------------------
# Load dataset
# ------------------------------
file_path = "/Users/veerababu/Downloads/master_list_cleaned.csv"
df = pd.read_csv(file_path)

print(f"Dataset loaded successfully: {file_path}\n")
print(f"Dataset Shape: {df.shape}\n")

# ------------------------------
# Column Info
# ------------------------------
print("Column Info:")
print(df.info())
print("\n")

# ------------------------------
# Summary Statistics
# ------------------------------
print("Summary Statistics:")
print(df.describe(include='all').T.head(10))  # show top 10 for readability
print("\n")

# ------------------------------
# Missing Values
# ------------------------------
print("Missing Values per Column:")
print(df.isnull().sum())
print("\n")

# ------------------------------
# Numeric Missing Check (Safe)
# ------------------------------
# Define possible numeric columns of interest
numeric_cols = ['total_injuries', 'total_days_missed']

# Keep only columns that actually exist in dataset
numeric_cols = [col for col in numeric_cols if col in df.columns]

if numeric_cols:
    print("Missing Values after numeric conversion:")
    print(df[numeric_cols].isnull().sum())
else:
    print("⚠️ No injury-related numeric columns found in this dataset.\n")

# ------------------------------
# Optional: Correlation matrix (if numeric features exist)
# ------------------------------
num_df = df.select_dtypes(include=['int64', 'float64'])
if not num_df.empty:
    print("Correlation Matrix (top 5 columns):")
    print(num_df.corr().iloc[:5, :5])
else:
    print("No numeric columns found for correlation matrix.")

print("\n✅ EDA completed successfully!")
