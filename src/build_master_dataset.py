import os
import logging
import pandas as pd
import numpy as np
import unicodedata

# ------------- CONFIG -------------
PROJECT_PATH = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-"
PROCESSED_DIR = os.path.join(PROJECT_PATH, "processed_data")

TRANSFERMARKT_PATH = os.path.join(PROCESSED_DIR, "transfermarkt_player_data.csv")
INJURY_PATH       = os.path.join(PROCESSED_DIR, "player_injury_features.csv")
PERFORMANCE_PATH  = os.path.join(PROCESSED_DIR, "player_performance.csv")
FINAL_DATA_PATH   = os.path.join(PROCESSED_DIR, "final_data.csv")

# inclusive season window
MIN_SEASON = 2019
MAX_SEASON = 2025

# essential columns that must exist in TM to keep a row
TM_REQUIRED = ["player_name", "season"]
# ---------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def normalize_name(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # collapse spaces
    s = " ".join(s.split())
    return s

def load_and_prepare_transfermarkt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only essential + everything else we need later
    # if our TM file has many columns, we keep them — just enforce name/season
    if "player_name" not in df.columns or "season" not in df.columns:
        raise ValueError("Transfermarkt file must contain 'player_name' and 'season' columns.")

    df["player_name"] = df["player_name"].astype(str)
    df["player_name_norm"] = df["player_name"].map(normalize_name)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    # filter seasons
    df = df[df["season"].between(MIN_SEASON, MAX_SEASON, inclusive="both")]

    # drop rows missing required
    df = df.dropna(subset=["player_name_norm", "season"])

    # drop duplicates by (player_name_norm, season) — keep first
    df = df.sort_values(["player_name_norm", "season"]).drop_duplicates(
        subset=["player_name_norm", "season"], keep="first"
    )

    # keep a clean external name column
    return df

def aggregate_injuries(df_inj: pd.DataFrame) -> pd.DataFrame:
    """
    Expect at least ['player_name','season'] in injury file.
    Aggregates to one row per player-season.
    """
    if "player_name" not in df_inj.columns or "season" not in df_inj.columns:
        raise ValueError("Injury file must contain 'player_name' and 'season' columns.")

    df = df_inj.copy()
    df["player_name_norm"] = df["player_name"].astype(str).map(normalize_name)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    # build aggregation dynamically based on available cols
    agg = {"player_name_norm": "first", "season": "first"}

    if "injury_days" in df.columns:
        agg["injury_days"] = "sum"
    if "matches_missed" in df.columns:
        agg["matches_missed"] = "sum"
    if "avg_rating_before_injury" in df.columns:
        agg["avg_rating_before_injury"] = "mean"
    if "avg_rating_after_injury" in df.columns:
        agg["avg_rating_after_injury"] = "mean"
    if "rating_drop" in df.columns:
        agg["rating_drop"] = "mean"

    # always add an injury_count = number of injury records that season
    df["injury_count"] = 1
    agg["injury_count"] = "sum"

    grouped = (
        df.groupby(["player_name_norm", "season"], as_index=False)
          .agg(agg)
    )

    return grouped


def prepare_performance(df_perf: pd.DataFrame) -> pd.DataFrame:
    if "player_name" not in df_perf.columns or "season" not in df_perf.columns:
        raise ValueError("Performance file must contain 'player_name' and 'season' columns.")
    df = df_perf.copy()
    df["player_name_norm"] = df["player_name"].astype(str).map(normalize_name)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    # If performance has multiple rows per player-season, reduce to one
    # Example: take sum for counting stats, mean for rates if present
    # Here we just take the first after sorting. Customize as needed.
    perf_cols = [c for c in df.columns if c not in {"player_name", "player_name_norm", "season"}]
    df = (
        df.sort_values(["player_name_norm", "season"])
          .drop_duplicates(subset=["player_name_norm", "season"], keep="first")
          .reset_index(drop=True)
    )
    return df

def main():
    logging.info("Loading Transfermarkt...")
    tm = load_and_prepare_transfermarkt(TRANSFERMARKT_PATH)

    # remove rows missing any required fields we care about
    if TM_REQUIRED:
        tm = tm.dropna(subset=[c for c in TM_REQUIRED if c in tm.columns])

    logging.info(f"Transfermarkt rows after filtering/dedup: {len(tm):,}")

    # --- Injury (LEFT join; NaN => assume no injury that season) ---
    if os.path.exists(INJURY_PATH):
        logging.info("Loading Injury data...")
        inj_raw = pd.read_csv(INJURY_PATH)
        inj = aggregate_injuries(inj_raw)
        tm_inj = tm.merge(
            inj.drop(columns=["player_name"], errors="ignore"),
            on=["player_name_norm", "season"],
            how="left",
            suffixes=("","_inj")
        )
        logging.info(f"After injury merge: {len(tm_inj):,} rows (left join)")
    else:
        logging.warning("Injury file not found — continuing without injuries.")
        tm_inj = tm.copy()

    # --- Performance (INNER join; drop players with no performance) ---
    if os.path.exists(PERFORMANCE_PATH):
        logging.info("Loading Performance data...")
        perf_raw = pd.read_csv(PERFORMANCE_PATH)
        perf = prepare_performance(perf_raw)
        # Prevent column collisions
        clash = set(tm_inj.columns) & set(perf.columns) - {"player_name_norm", "season"}
        perf = perf.rename(columns={c: f"{c}_perf" for c in clash})

        final = tm_inj.merge(
            perf.drop(columns=["player_name"], errors="ignore"),
            on=["player_name_norm", "season"],
            how="inner"
        )
        logging.info(f"After performance merge (inner): {len(final):,} rows")
    else:
        raise FileNotFoundError("Performance file not found; required for final_data.")

    # --- Final cleaning ---
    # Drop 100% duplicate rows
    final = final.drop_duplicates()

    # Keep the original player_name for display, ensure it's present
    if "player_name" not in final.columns and "player_name" in tm.columns:
        final["player_name"] = final["player_name"]

    # we keep NaN in injury columns to encode “no injury”.
    # But we can enforce no nulls in essential keys:
    final = final.dropna(subset=["player_name_norm", "season"])

    # Save
    os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
    final.to_csv(FINAL_DATA_PATH, index=False)
    logging.info(f"Saved final_data to: {FINAL_DATA_PATH}")
    logging.info(final.head(3).to_string())

if __name__ == "__main__":
    main()