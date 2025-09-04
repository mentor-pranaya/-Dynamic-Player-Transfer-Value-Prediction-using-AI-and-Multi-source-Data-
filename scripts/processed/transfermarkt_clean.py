import pandas as pd
import os

# === Paths ===
BASE_DIR = "/content/drive/MyDrive/player_value_prediction_project/data"
RAW_PATH = os.path.join(BASE_DIR, "raw", "player_scores")
OUT_PATH = os.path.join(BASE_DIR, "processed", "transfermarkt_clean.csv")


def load_data():
    """Load raw Transfermarkt datasets"""
    players = pd.read_csv(os.path.join(RAW_PATH, "players.csv"))
    players = players.rename(columns={"name": "player_name"})
    valuations = pd.read_csv(os.path.join(RAW_PATH, "player_valuations.csv"))
    transfers = pd.read_csv(os.path.join(RAW_PATH, "transfers.csv"))
    appearances = pd.read_csv(os.path.join(RAW_PATH, "appearances.csv"))

    print("Loading shapes:")
    print(" players:", players.shape)
    print(" valuations:", valuations.shape)
    print(" transfers:", transfers.shape)
    print(" appearances:", appearances.shape)

    return players, valuations, transfers, appearances


def preprocess(players, valuations, transfers, appearances):
    """Clean and aggregate Transfermarkt datasets"""

    # --- Standardize player names ---
    for df in (players, transfers, appearances):
        if "player_name" in df.columns:
            df["player_name"] = df["player_name"].astype(str).str.strip().str.title()

    # --- Convert dates ---
    valuations["valuation_date"] = pd.to_datetime(valuations["date"], errors="coerce")
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

    # --- Seasons ---
    valuations["season"] = valuations["valuation_date"].dt.year
    transfers["season"] = transfers["transfer_date"].dt.year
    appearances["season"] = appearances["date"].dt.year

    # --- Drop rows with missing merge keys ---
    valuations = valuations.dropna(subset=["player_id", "season"])
    transfers = transfers.dropna(subset=["player_id", "season"])
    appearances = appearances.dropna(subset=["player_id", "season"])

    # --- Aggregate Valuations ---
    val_agg = valuations.groupby(["player_id", "season"], as_index=False).agg(
        avg_market_value=("market_value_in_eur", "mean"),
        max_market_value=("market_value_in_eur", "max"),
        min_market_value=("market_value_in_eur", "min"),
        n_valuations=("market_value_in_eur", "count"),
    )

    # --- Transfers (fallback to transfer's own MV if no valuation) ---
    transfers["transfer_premium"] = transfers["transfer_fee"] - transfers["market_value_in_eur"]

    trans_agg = transfers.groupby(["player_id", "season"], as_index=False).agg(
        num_transfers=("transfer_date", "count"),
        avg_transfer_fee=("transfer_fee", "mean"),
        last_transfer_fee=("transfer_fee", "last"),
        avg_transfer_premium=("transfer_premium", "mean"),
        last_transfer_premium=("transfer_premium", "last"),
        last_transfer_club_to=("to_club_name", "last"),
        last_transfer_club_from=("from_club_name", "last"),
    )

    # --- Appearances ---
    apps_agg = appearances.groupby(["player_id", "season"], as_index=False).agg(
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_minutes_played=("minutes_played", "sum"),
        total_yellow_cards=("yellow_cards", "sum"),
        total_red_cards=("red_cards", "sum"),
        matches_played=("game_id", "count"),
    )

    return players, val_agg, trans_agg, apps_agg


def merge_data(players, val_agg, trans_agg, apps_agg):
    """Merge valuation + transfer + appearance + player info"""

    merged = val_agg.merge(trans_agg, on=["player_id", "season"], how="outer")
    merged = merged.merge(apps_agg, on=["player_id", "season"], how="outer")

    # --- Add player metadata ---
    player_info = players[[
        "player_id", "player_name", "last_season", "current_club_id", "country_of_birth",
        "country_of_citizenship", "date_of_birth", "sub_position", "position", "foot",
        "height_in_cm", "contract_expiration_date", "agent_name", "current_club_name",
        "highest_market_value_in_eur"
    ]].drop_duplicates()

    final_df = merged.merge(player_info, on="player_id", how="left")

    # ✅ Keep only one player_name column
    if "player_name_x" in final_df.columns and "player_name_y" in final_df.columns:
        final_df["player_name"] = final_df["player_name_y"].fillna(final_df["player_name_x"])
        final_df = final_df.drop(columns=["player_name_x", "player_name_y"])
    elif "player_name_x" in final_df.columns:
        final_df = final_df.rename(columns={"player_name_x": "player_name"})
    elif "player_name_y" in final_df.columns:
        final_df = final_df.rename(columns={"player_name_y": "player_name"})

    # --- Reorder ---
    cols = ["player_id", "player_name", "season"] + [
        c for c in final_df.columns if c not in ["player_id", "player_name", "season"]
    ]
    final_df = final_df[cols].sort_values(by=["player_name", "season"])

    return final_df


def main():
    players, valuations, transfers, appearances = load_data()
    players, val_agg, trans_agg, apps_agg = preprocess(players, valuations, transfers, appearances)
    final_df = merge_data(players, val_agg, trans_agg, apps_agg)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved Transfermarkt cleaned dataset to {OUT_PATH}")
    print("Preview:")
    print(final_df.head())


if __name__ == "__main__":
    main()
