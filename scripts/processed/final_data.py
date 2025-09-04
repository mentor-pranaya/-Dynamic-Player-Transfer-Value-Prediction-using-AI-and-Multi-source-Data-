import pandas as pd
import os

# === Paths ===
BASE_DIR = "/content/drive/MyDrive/player_value_prediction_project/data"
TRANSFERMARKT_PATH = os.path.join(BASE_DIR, "processed", "transfermarkt_clean.csv")
INJURY_PATH        = os.path.join(BASE_DIR, "processed", "injury_clean.csv")
STATSBOMB_PATH     = os.path.join(BASE_DIR, "processed", "statsbomb_clean.csv")
FINAL_OUT          = os.path.join(BASE_DIR, "processed", "final_data.csv")

# === Helper: Normalize player names ===
def normalize_name(s):
    if pd.isna(s):
        return None
    return str(s).strip().lower()

def main():
    print("Loading datasets...")
    tm = pd.read_csv(TRANSFERMARKT_PATH, low_memory=False)
    inj = pd.read_csv(INJURY_PATH, low_memory=False)
    sb  = pd.read_csv(STATSBOMB_PATH, low_memory=False)

    print(f"Transfermarkt: {tm.shape}, Injury: {inj.shape}, StatsBomb: {sb.shape}")

    # --- Normalize keys ---
    for df in (tm, inj, sb):
        df["player_name_norm"] = df["player_name"].map(normalize_name)
        # force season to string for consistent joins
        df["season"] = df["season"].astype(str)

    # --- Merge datasets ---
    print("Merging datasets...")
    merged = tm.merge(
        inj.drop(columns=["player_name"], errors="ignore"),
        on=["player_name_norm", "season"],
        how="left",
        suffixes=("", "_inj")
    )
    print("After injury merge:", merged.shape)

    merged = merged.merge(
        sb.drop(columns=["player_name"], errors="ignore"),
        on=["player_name_norm", "season"],
        how="left",
        suffixes=("", "_sb")
    )
    print("After statsbomb merge:", merged.shape)

    # --- Handle missing values safely ---
    # Numeric columns
    numeric_cols = merged.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        merged[col] = merged[col].fillna(0)

    # Datetime columns
    datetime_cols = merged.select_dtypes(include=["datetime64[ns]"]).columns
    for col in datetime_cols:
        merged[col] = merged[col].fillna(pd.NaT)

    # Object/string columns
    categorical_cols = merged.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        merged[col] = merged[col].fillna("Unknown")

    # --- Final clean-up ---
    merged = merged.drop_duplicates(subset=["player_name_norm", "season"])
    merged = merged.reset_index(drop=True)

    # --- Save ---
    os.makedirs(os.path.dirname(FINAL_OUT), exist_ok=True)
    merged.to_csv(FINAL_OUT, index=False)

    print(f"✅ Final dataset saved to {FINAL_OUT}")
    print("Preview:")
    print(merged.head())

if __name__ == "__main__":
    main()
