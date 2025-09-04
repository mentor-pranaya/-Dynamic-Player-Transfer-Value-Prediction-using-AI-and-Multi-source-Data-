# Create directory
!mkdir -p ./data/raw/injuries/

# Download and unzip
!kaggle datasets download -d amritbiswas007/player-injuries-and-team-performance-dataset -p ./data/raw/injuries/
!unzip -q ./data/raw/injuries/player-injuries-and-team-performance-dataset.zip -d ./data/raw/injuries/

import os
print("Files in injury dataset:", os.listdir("./data/raw/injuries/"))

import pandas as pd

injuries = pd.read_csv("./data/raw/injuries/player_injuries_impact.csv")
print("Shape:", injuries.shape)
print("Columns:", injuries.columns.tolist())
print(injuries.head())

# src/extract_injuries.py
import pandas as pd
import os

def load_injury_data(input_path="./data/raw/injuries/player_injuries_impact.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"✅ Injury dataset loaded: {df.shape}")
    return df

def preprocess_injury_data(df):
    # Convert dates
    df["Date of Injury"] = pd.to_datetime(df["Date of Injury"], errors="coerce")
    df["Date of return"] = pd.to_datetime(df["Date of return"], errors="coerce")

    # Injury duration
    df["injury_duration_days"] = (df["Date of return"] - df["Date of Injury"]).dt.days

    # Matches missed
    missed_cols = [c for c in df.columns if "missed_match" in c]
    df["matches_missed"] = df[missed_cols].apply(lambda row: row.ne("N.A.").sum(), axis=1)

    # Avg ratings (before vs after injury)
    def avg_rating(cols):
        vals = []
        for v in cols:
            try:
                vals.append(float(str(v).replace("(S)", "")))
            except:
                continue
        return sum(vals) / len(vals) if vals else None

    before_cols = [c for c in df.columns if "before_injury_Player_rating" in c]
    after_cols  = [c for c in df.columns if "after_injury_Player_rating" in c]

    df["avg_rating_before"] = df[before_cols].apply(avg_rating, axis=1)
    df["avg_rating_after"]  = df[after_cols].apply(avg_rating, axis=1)
    df["rating_change"] = df["avg_rating_after"] - df["avg_rating_before"]

    return df

if __name__ == "__main__":
    df = load_injury_data()
    df = preprocess_injury_data(df)
    print(df[["Name", "Injury", "injury_duration_days", "matches_missed", "rating_change"]].head())

