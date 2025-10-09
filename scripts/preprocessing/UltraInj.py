import pandas as pd
import unicodedata
import os

# -----------------------------
# 1. File locations
# -----------------------------
file_injuries_master = "/Users/veerababu/Downloads/cleaned/injuries_master_aggregated_full.csv"
output_file = "/Users/veerababu/Downloads/cleaned/injuries_master_cleaned_final_sorted.csv"

# -----------------------------
# 2. Check if file exists
# -----------------------------
if not os.path.exists(file_injuries_master):
    raise FileNotFoundError(f"The file does not exist: {file_injuries_master}")

# -----------------------------
# 3. Load CSV
# -----------------------------
df = pd.read_csv(file_injuries_master)

# -----------------------------
# 4. Clean column names
# -----------------------------
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# 5. Remove duplicate rows
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# 6. Remove rows with missing values in important columns
# -----------------------------
important_cols = ['player', 'total_days_missed', 'total_injuries']
df = df.dropna(subset=important_cols)

# -----------------------------
# 7. Remove accents from player names
# -----------------------------
def remove_accents(text):
    if isinstance(text, str):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if ord(c) < 128)
    return text

df['player'] = df['player'].apply(remove_accents)

# -----------------------------
# 8. Sort alphabetically by player name
# -----------------------------
df = df.sort_values(by='player')

# -----------------------------
# 9. Reset index
# -----------------------------
df = df.reset_index(drop=True)

# -----------------------------
# 10. Save cleaned and sorted data
# -----------------------------
df.to_csv(output_file, index=False)

print(f"✅ Injuries master file cleaned, sorted alphabetically, and saved: {output_file}")
print(df.head(20))
