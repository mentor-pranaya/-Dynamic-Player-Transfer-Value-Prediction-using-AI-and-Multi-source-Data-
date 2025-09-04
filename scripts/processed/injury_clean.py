import pandas as pd
import os
import re
import numpy as np

# === Paths ===
BASE_DIR = "/content/drive/MyDrive/player_value_prediction_project/data"
RAW_FILE = os.path.join(BASE_DIR, "raw", "injuries", "player_injuries_impact.csv")
OUT_FILE = os.path.join(BASE_DIR, "processed", "injury_clean.csv")


def clean_rating(value):
    """Convert messy ratings (e.g., '6(S)', 'N.A.') into floats."""
    if pd.isna(value):
        return np.nan
    val = str(value)

    # Remove unwanted markers
    val = val.replace("N.A.", "").replace("(S)", "")

    # Keep only digits and dots
    val = re.sub(r"[^0-9.]", "", val)

    # Handle cases with multiple dots (take first float-like part)
    parts = val.split(".")
    if len(parts) > 2:
        try:
            return float(parts[0] + "." + parts[1])
        except:
            return np.nan

    try:
        return float(val)
    except:
        return np.nan


def process_injuries():
    df = pd.read_csv(RAW_FILE)

    # --- Rename core columns ---
    df.rename(columns={
        "Name": "player_name",
        "Team Name": "team_name",
        "Position": "position",
        "Age": "age",
        "FIFA rating": "fifa_rating",
        "Season": "season"
    }, inplace=True)

    # --- Standardize player names ---
    df["player_name"] = df["player_name"].astype(str).str.strip().str.title()

    # --- Fix season format (2019/20 → 2019/2020) ---
    df["season"] = df["season"].astype(str).str.replace(
        r"(\d{4})/(\d{2})",
        lambda m: f"{m.group(1)}/20{m.group(2)}",
        regex=True
    )

    # --- Dates ---
    df["Date of Injury"] = pd.to_datetime(df["Date of Injury"], errors="coerce")
    df["Date of return"] = pd.to_datetime(df["Date of return"], errors="coerce")

    # --- Injury duration ---
    df["injury_days"] = (df["Date of return"] - df["Date of Injury"]).dt.days

    # --- Matches missed ---
    missed_cols = [c for c in df.columns if "missed_match_Result" in c]
    df["matches_missed"] = df[missed_cols].notna().sum(axis=1)

    # --- Ratings before/after injury ---
    before_cols = [c for c in df.columns if "before_injury_Player_rating" in c]
    after_cols = [c for c in df.columns if "after_injury_Player_rating" in c]

    for col in before_cols + after_cols:
        df[col] = df[col].apply(clean_rating)

    df["avg_rating_before_injury"] = df[before_cols].mean(axis=1)
    df["avg_rating_after_injury"] = df[after_cols].mean(axis=1)
    df["rating_drop"] = df["avg_rating_before_injury"] - df["avg_rating_after_injury"]

    # --- Final clean dataset ---
    final_df = df[[
        "player_name", "season", "team_name", "position", "age", "fifa_rating",
        "injury_days", "matches_missed",
        "avg_rating_before_injury", "avg_rating_after_injury", "rating_drop"
    ]]

    # --- Aggregate if multiple injuries in same season ---
    final_df = final_df.groupby(["player_name", "season"], as_index=False).agg({
        "team_name": "first",
        "position": "first",
        "age": "mean",
        "fifa_rating": "mean",
        "injury_days": "sum",
        "matches_missed": "sum",
        "avg_rating_before_injury": "mean",
        "avg_rating_after_injury": "mean",
        "rating_drop": "mean"
    })

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    final_df.to_csv(OUT_FILE, index=False)

    print(f"✅ Saved injury dataset to {OUT_FILE}")
    print("Preview:")
    print(final_df.head())


if __name__ == "__main__":
    process_injuries()
