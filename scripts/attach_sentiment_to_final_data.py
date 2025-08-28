import os
import logging
import pandas as pd

# Paths
PROJECT_PATH  = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-"
PROCESSED_DIR = os.path.join(PROJECT_PATH, "processed_data")

FINAL_DATA    = os.path.join(PROCESSED_DIR, "final_data.csv")
SENTIMENT     = os.path.join(PROCESSED_DIR, "reddit_sentiment.csv")
FINAL_WITH_S  = os.path.join(PROCESSED_DIR, "final_data_with_sentiment.csv")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    # Load datasets
    df = pd.read_csv(FINAL_DATA)
    se = pd.read_csv(SENTIMENT)

    # Standardize merge keys
    for data in (df, se):
        data["player_name"] = data["player_name"].astype(str).str.strip().str.lower()
        data["season"] = data["season"].astype(str).str.strip()

    # Avoid column collisions
    clash = set(df.columns) & (set(se.columns) - {"player_name","season"})
    se = se.rename(columns={c: f"reddit_{c}" for c in clash})

    # Merge
    out = df.merge(se, on=["player_name","season"], how="left")

    # Save
    out.to_csv(FINAL_WITH_S, index=False)
    logging.info(f"Saved merged dataset: {FINAL_WITH_S}")

    # Log merge stats
    sentiment_cols = [c for c in out.columns if c.startswith("reddit_")]
    if sentiment_cols:
        merged_count = out[sentiment_cols[0]].notna().sum()
        logging.info(f"Sentiment data merged for {merged_count}/{len(out)} player-seasons")

    # Warn about unmatched sentiment rows
    missing = se[~se[["player_name","season"]].apply(tuple, 1).isin(
                df[["player_name","season"]].apply(tuple, 1))]
    if not missing.empty:
        logging.warning(f" {len(missing)} sentiment rows did not match any player-season in final_data")

    # Show preview
    logging.info("Preview of merged data:")
    logging.info(out.head(5).to_string())

if __name__ == "__main__":
    main()
