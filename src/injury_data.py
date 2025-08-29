import pandas as pd
import os
import re
import numpy as np

# Paths
BASE_DIR = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-"
INPUT_FILE = os.path.join(BASE_DIR, "data\historical_Injury\player_injuries_impact.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "processed_data", "player_injury_features.csv")


def clean_rating(value):
    """Clean messy rating strings into numeric floats."""
    if pd.isna(value):
        return np.nan
    val = str(value)

    # Remove 'N.A.' and '(S)'
    val = val.replace("N.A.", "").replace("(S)", "")

    # Keep only digits and dots
    val = re.sub(r"[^0-9.]", "", val)

    # Handle multiple dots like "7.46.15.8" -> "7.46"
    parts = val.split(".")
    if len(parts) > 2:
        try:
            return float(parts[0] + "." + parts[1])  # take first two parts
        except:
            return np.nan

    # Convert to float safely
    try:
        return float(val)
    except:
        return np.nan


def process_injuries(INPUT_FILE, OUTPUT_FILE):
    # Load CSV
    df = pd.read_csv(INPUT_FILE)

    # Rename important columns
    df.rename(columns={'Name': 'player_name',
                       'Team Name': 'team_name',
                       'Position': 'position',
                       'Age': 'age',
                       'FIFA rating': 'fifa_rating'}, inplace=True)

    # Standardize player names
    df['player_name'] = df['player_name'].astype(str).str.strip().str.title()

    # Standardize season format (2019/20 -> 2019/2020)
    df['season'] = df['season'].astype(str).str.replace(
        r'(\d{4})/(\d{2})',
        lambda m: f"{m.group(1)}/20{m.group(2)}",
        regex=True
    )

    # Convert dates
    df['Date of Injury'] = pd.to_datetime(df['Date of Injury'], errors='coerce')
    df['Date of return'] = pd.to_datetime(df['Date of return'], errors='coerce')

    # Injury duration (days)
    df['injury_days'] = (df['Date of return'] - df['Date of Injury']).dt.days

    # Matches missed count
    missed_cols = [c for c in df.columns if "missed_match_Result" in c]
    df['matches_missed'] = df[missed_cols].notna().sum(axis=1)

    # Avg rating before injury
    before_cols = [c for c in df.columns if "before_injury_Player_rating" in c]
    for col in before_cols:
        df[col] = df[col].apply(clean_rating)
    df['avg_rating_before_injury'] = df[before_cols].mean(axis=1)

    # Avg rating after injury
    after_cols = [c for c in df.columns if "after_injury_Player_rating" in c]
    for col in after_cols:
        df[col] = df[col].apply(clean_rating)
    df['avg_rating_after_injury'] = df[after_cols].mean(axis=1)

    # Rating drop
    df['rating_drop'] = df['avg_rating_before_injury'] - df['avg_rating_after_injury']

    # Keep only useful columns
    final_df = df[['player_name', 'season', 'team_name', 'position', 'age', 'fifa_rating',
                   'injury_days', 'matches_missed',
                   'avg_rating_before_injury', 'avg_rating_after_injury', 'rating_drop']]

    # Aggregate if multiple injuries per player-season
    final_df = final_df.groupby(['player_name', 'season'], as_index=False).agg({
        'team_name': 'first',
        'position': 'first',
        'age': 'mean',
        'fifa_rating': 'mean',
        'injury_days': 'sum',
        'matches_missed': 'sum',
        'avg_rating_before_injury': 'mean',
        'avg_rating_after_injury': 'mean',
        'rating_drop': 'mean'
    })

    # Save cleaned dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Processed file saved as {OUTPUT_FILE}")
    print(final_df.head())


if __name__ == "__main__":
    process_injuries(INPUT_FILE, OUTPUT_FILE)