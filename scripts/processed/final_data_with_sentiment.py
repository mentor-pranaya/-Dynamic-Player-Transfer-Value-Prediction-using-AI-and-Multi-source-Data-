import os
import pandas as pd

# === Paths ===
BASE_DIR = "/content/drive/MyDrive/player_value_prediction_project/data"
FINAL_DATA_PATH = os.path.join(BASE_DIR, "processed", "final_data.csv")
SENTIMENT_PATH = os.path.join(BASE_DIR, "raw", "final_reddit_sentiment.csv")
OUT_PATH = os.path.join(BASE_DIR, "processed", "final_data_with_sentiment.csv")

def main():
    print("Loading datasets...")
    df = pd.read_csv(FINAL_DATA_PATH, low_memory=False)
    se = pd.read_csv(SENTIMENT_PATH, low_memory=False)

    print(f"Final data: {df.shape}, Sentiment: {se.shape}")

    # --- Normalize merge keys ---
    df["player_name_norm"] = df["player_name"].astype(str).str.strip().str.lower()
    df["season_norm"] = df["season"].astype(str).str.strip()

    se["player_name_norm"] = se["player_name"].astype(str).str.strip().str.lower()
    se["season_norm"] = se["season"].astype(str).str.strip()

    # --- Prefix ALL sentiment columns (except keys) ---
    sentiment_cols = [c for c in se.columns if c not in ["player_name", "season", "player_name_norm", "season_norm"]]
    se = se.rename(columns={c: f"reddit_{c}" for c in sentiment_cols})

    # --- Merge ---
    merged = df.merge(
        se.drop(columns=["player_name","season"], errors="ignore"),
        on=["player_name_norm","season_norm"],
        how="left"
    )

    # Drop helper keys
    merged = merged.drop(columns=["player_name_norm","season_norm"])

    # --- Add sentiment_missing_flag ---
    merged["sentiment_missing_flag"] = merged["reddit_num_posts"].isna().astype(int)

    # --- Handle missing values with defaults ---
    fill_defaults = {
        "reddit_num_posts": 0,
        "reddit_num_comments_used": 0,
        "reddit_pos_ratio": 0.0,
        "reddit_neu_ratio": 0.0,
        "reddit_neg_ratio": 0.0,
        "reddit_mean_compound": 0.0,
        "reddit_fallback_used": 1,
        "reddit_subreddits_covered": "None",
    }

    for col, default in fill_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)

    # --- Save ---
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved final dataset with sentiment to {OUT_PATH}")
    print("Preview:")
    print(merged.head())

if __name__ == "__main__":
    main()
