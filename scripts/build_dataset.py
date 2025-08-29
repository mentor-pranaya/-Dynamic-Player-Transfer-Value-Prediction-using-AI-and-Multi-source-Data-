import os
import logging
import pandas as pd
import numpy as np
import unicodedata

# ------------------ CONFIG ------------------
PROJECT_PATH = r"C:\Users\Abhinav\Desktop\Project"
PROCESSED_DIR = os.path.join(PROJECT_PATH, "processed_data")

TRANSFERMARKT_PATH = os.path.join(PROCESSED_DIR, "transfermarkt_player_data.csv")
INJURY_PATH       = os.path.join(PROCESSED_DIR, "player_injury_features.csv")
PERFORMANCE_PATH  = os.path.join(PROCESSED_DIR, "player_performance.csv")
FINAL_DATA_PATH   = os.path.join(PROCESSED_DIR, "final_data.csv")

SEASON_START = 2019
SEASON_END   = 2025

REQUIRED_COLS_TM = ["player_name", "season"]
# ------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
np.random.seed(42)  # reproducibility

# ---------- Helpers ----------
def clean_name(name: str) -> str:
    """Normalize player names (lowercase, remove accents, trim spaces)."""
    if pd.isna(name):
        return name
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    return " ".join(name.split())


def load_transfermarkt(path: str) -> pd.DataFrame:
    """Load and filter Transfermarkt data."""
    df = pd.read_csv(path)

    if not all(c in df.columns for c in REQUIRED_COLS_TM):
        raise ValueError("Transfermarkt file missing required columns.")

    df["player_name_norm"] = df["player_name"].map(clean_name)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    # filter valid seasons
    df = df[df["season"].between(SEASON_START, SEASON_END, inclusive="both")]

    # drop invalid rows & duplicates
    df = df.dropna(subset=["player_name_norm", "season"])
    df = df.sort_values(["player_name_norm", "season"]).drop_duplicates(
        subset=["player_name_norm", "season"], keep="first"
    )

    return df


def summarize_injuries(df_inj: pd.DataFrame) -> pd.DataFrame:
    """Aggregate injury stats to one row per player-season."""
    if not {"player_name", "season"}.issubset(df_inj.columns):
        raise ValueError("Injury file missing required columns.")

    df_inj["player_name_norm"] = df_inj["player_name"].map(clean_name)
    df_inj["season"] = pd.to_numeric(df_inj["season"], errors="coerce").astype("Int64")
    df_inj["injury_count"] = 1

    agg_map = {
        "player_name_norm": "first",
        "season": "first",
        "injury_count": "sum"
    }
    if "injury_days" in df_inj.columns:
        agg_map["injury_days"] = "sum"
    if "matches_missed" in df_inj.columns:
        agg_map["matches_missed"] = "sum"
    if "avg_rating_before_injury" in df_inj.columns:
        agg_map["avg_rating_before_injury"] = "mean"
    if "avg_rating_after_injury" in df_inj.columns:
        agg_map["avg_rating_after_injury"] = "mean"

    return df_inj.groupby(["player_name_norm", "season"], as_index=False).agg(agg_map)


def prepare_performance(df_perf: pd.DataFrame) -> pd.DataFrame:
    """Reduce performance data to one row per player-season."""
    if not {"player_name", "season"}.issubset(df_perf.columns):
        raise ValueError("Performance file missing required columns.")

    df_perf["player_name_norm"] = df_perf["player_name"].map(clean_name)
    df_perf["season"] = pd.to_numeric(df_perf["season"], errors="coerce").astype("Int64")

    # keep one row per player-season (first after sorting)
    df_perf = (
        df_perf.sort_values(["player_name_norm", "season"])
               .drop_duplicates(subset=["player_name_norm", "season"], keep="first")
               .reset_index(drop=True)
    )
    return df_perf


# ---------- Main Pipeline ----------
def main():
    logging.info("📂 Reading Transfermarkt dataset...")
    df_tm = load_transfermarkt(TRANSFERMARKT_PATH)
    logging.info(f"✅ Transfermarkt rows after filtering: {len(df_tm):,}")

    # --- Injuries (LEFT JOIN) ---
    if os.path.exists(INJURY_PATH):
        logging.info("📂 Reading Injury dataset...")
        df_inj = pd.read_csv(INJURY_PATH)
        df_injury = summarize_injuries(df_inj)
        df_tm = df_tm.merge(
            df_injury.drop(columns=["player_name"], errors="ignore"),
            on=["player_name_norm", "season"],
            how="left"
        )
        logging.info(f"➡️ After injury merge: {len(df_tm):,} rows")
    else:
        logging.warning("⚠️ Injury file not found. Skipping injuries.")

    # --- Performance (INNER JOIN) ---
    if os.path.exists(PERFORMANCE_PATH):
        logging.info("📂 Reading Performance dataset...")
        df_perf = pd.read_csv(PERFORMANCE_PATH)
        df_perf = prepare_performance(df_perf)

        # handle duplicate columns
        clash = set(df_tm.columns) & set(df_perf.columns) - {"player_name_norm", "season"}
        df_perf = df_perf.rename(columns={c: f"{c}_perf" for c in clash})

        df_final = df_tm.merge(
            df_perf.drop(columns=["player_name"], errors="ignore"),
            on=["player_name_norm", "season"],
            how="inner"
        )
        logging.info(f"➡️ After performance merge: {len(df_final):,} rows")
    else:
        raise FileNotFoundError("Performance dataset is required but missing.")

    # --- Final cleanup ---
    df_final = df_final.drop_duplicates()
    df_final = df_final.dropna(subset=["player_name_norm", "season"])

    os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
    df_final.to_csv(FINAL_DATA_PATH, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Master dataset saved at: {FINAL_DATA_PATH}")
    logging.info(df_final.head(5).to_string())


if __name__ == "__main__":
    main()
