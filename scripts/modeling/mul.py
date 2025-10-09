import pandas as pd
import numpy as np
import os

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "/Users/veerababu/Downloads/master_list_cleaned.csv"
OUTPUT_FILE = "/Users/veerababu/Desktop/sample_master_list.csv"
TIME_STEPS = 12  # Number of rows per player
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# -----------------------------
# Load original master CSV
# -----------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Original data shape: {df.shape}")

# -----------------------------
# Identify numeric and categorical columns
# -----------------------------
NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = df.select_dtypes(include=['object']).columns.tolist()

# Columns we don't want to change
IGNORE_COLS = ['player', 'player_id', 'slug', 'url', 'image_url']

# -----------------------------
# Expand each player to TIME_STEPS
# -----------------------------
rows = []
for _, row in df.iterrows():
    for t in range(1, TIME_STEPS+1):
        new_row = row.copy()
        new_row['time_step'] = t
        
        # Slightly modify numeric columns to simulate progression
        for col in NUMERIC_COLS:
            if col in IGNORE_COLS:
                continue
            # Linear progression
            new_row[col] = new_row[col] + (t-1)*new_row[col]*0.02
            # Add small noise (ensure scale is positive)
            scale = max(abs(new_row[col]*0.01), 1e-6)
            new_row[col] += np.random.normal(0, scale)
        
        rows.append(new_row)

# -----------------------------
# Create new DataFrame
# -----------------------------
df_expanded = pd.DataFrame(rows)

# Optional: shuffle rows
df_expanded = df_expanded.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# -----------------------------
# Save reconstructed master CSV
# -----------------------------
df_expanded.to_csv(OUTPUT_FILE, index=False)
print(f"Sample master CSV created at: {OUTPUT_FILE}")
print(f"New data shape: {df_expanded.shape}")
