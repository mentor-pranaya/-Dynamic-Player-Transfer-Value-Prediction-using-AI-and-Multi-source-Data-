import pandas as pd
import numpy as np


# Load dataset (CSV version)

file_path = r"C:\Users\samar\OneDrive\Documents\Infosys_internship\Player_Features_2022_23.csv"
df = pd.read_csv(file_path, encoding="utf-8")


# Clean column names

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%", "pct")

# Ensure 90s column exists
if "90s" not in df.columns:
    raise ValueError("Expected a '90s' column to normalize stats per 90 mins.")


# Per-90 Features

per90_features = {
    "gls": "goals_per90",
    "ast": "assists_per90",
    "sh": "shots_per90",
    "sot": "shots_on_target_per90",
    "tkl": "tackles_per90",
    "int": "interceptions_per90",
    "press": "pressures_per90",
    "blocks": "blocks_per90",
    "touches": "touches_per90",
    "passes_completed": "passes_completed_per90",
    "passes": "passes_attempted_per90"
}

for col, new_col in per90_features.items():
    if col in df.columns:
        df[new_col] = df[col] / df["90s"].replace(0, np.nan)


# Composite Features

if {"gls", "ast"}.issubset(df.columns):
    df["goal_contrib_per90"] = (df["gls"] + df["ast"]) / df["90s"].replace(0, np.nan)

if {"sot", "sh"}.issubset(df.columns):
    df["shot_accuracy"] = df["sot"] / df["sh"].replace(0, np.nan)

if {"passes_completed", "passes"}.issubset(df.columns):
    df["pass_accuracy"] = df["passes_completed"] / df["passes"].replace(0, np.nan)

# Defensive contribution index
if {"tkl", "int", "blocks"}.issubset(df.columns):
    df["defensive_index"] = (df["tkl"] + df["int"] + df["blocks"]) / df["90s"].replace(0, np.nan)


# Position Encoding

def map_position(pos):
    if not isinstance(pos, str):
        return "Other"
    pos = pos.upper()
    if pos.startswith("F") or pos in ["RW", "LW", "CF", "ST"]:
        return "Attacker"
    elif pos.startswith("M"):
        return "Midfielder"
    elif pos.startswith("D"):
        return "Defender"
    elif pos in ["GK", "G"]:
        return "Goalkeeper"
    else:
        return "Other"

if "pos" in df.columns:
    df["position_group"] = df["pos"].apply(map_position)
    df = pd.get_dummies(df, columns=["position_group"], prefix="pos")


# Age Features

if "age" in df.columns:
    df["age_squared"] = df["age"] ** 2
    df["age_bucket"] = pd.cut(df["age"], bins=[15, 20, 24, 28, 32, 36, 40, 50],
                              labels=["<20","20-24","25-28","29-32","33-36","37-40","40+"])


# Save engineered dataset

output_file = r"C:\Users\samar\OneDrive\Documents\Infosys_internship\Player_Features_2022_23.csv"
df.to_csv(output_file, index=False)

print(f"✅ Feature engineering completed. Saved to {output_file}, shape={df.shape}")
