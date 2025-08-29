import pandas as pd
import os


def load_data():
    """Load raw Transfermarkt data"""
    base_path = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-\data\transfermarkt\raw"

    players = pd.read_csv(os.path.join(base_path, "players.csv"))
    valuations = pd.read_csv(os.path.join(base_path, "player_valuations.csv"))
    transfers = pd.read_csv(os.path.join(base_path, "transfers.csv"))
    appearances = pd.read_csv(os.path.join(base_path, "appearances.csv"))

    return players, valuations, transfers, appearances


def preprocess(players, valuations, transfers, appearances):
    """Preprocess datasets for player_name + season level aggregation"""

    # Standardize player names
    players["player_name"] = players["name"].astype(str).str.strip().str.title()
    transfers["player_name"] = transfers["player_name"].astype(str).str.strip().str.title()
    appearances["player_name"] = appearances["player_name"].astype(str).str.strip().str.title()

    # Convert dates
    valuations["valuation_date"] = pd.to_datetime(valuations["date"], errors="coerce", dayfirst=True)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce", dayfirst=True)
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce", dayfirst=True)

    # Derive season from dates
    valuations["season"] = valuations["valuation_date"].dt.year
    transfers["season"] = transfers["transfer_date"].dt.year
    appearances["season"] = appearances["date"].dt.year

    # ---- Aggregate Valuations ----
    val_agg = valuations.groupby(["player_id", "season"], as_index=False).agg(
        avg_market_value=("market_value_in_eur", "mean"),
        max_market_value=("market_value_in_eur", "max"),
        min_market_value=("market_value_in_eur", "min"),
    )

    # ---- Transfer Premium (using merge_asof) ----
    valuations_sorted = valuations.sort_values(["player_id", "valuation_date"])
    transfers_sorted = transfers.sort_values(["player_id", "transfer_date"])

    transfers_with_val = pd.merge_asof(
        transfers_sorted.sort_values("transfer_date"),
        valuations_sorted.sort_values("valuation_date")[["player_id", "valuation_date", "market_value_in_eur"]],
        by="player_id",
        left_on="transfer_date",
        right_on="valuation_date",
        direction="backward"
    )

    # Decide which column to use as fallback market value in transfers
    transfer_mv_col = None
    for col in ["market_value_in_eur", "transfer_market_value"]:
        if col in transfers.columns:
            transfer_mv_col = col
            break

    # Final market value = nearest valuation (y), fallback to transfer's own (x)
    transfers_with_val["final_market_value"] = (
        transfers_with_val["market_value_in_eur_y"]
        .fillna(transfers_with_val["market_value_in_eur_x"])
    )

    # Compute premium
    transfers_with_val["transfer_premium"] = (
            transfers_with_val["transfer_fee"] - transfers_with_val["final_market_value"]
    )

    # ---- Aggregate Transfers ----
    trans_agg = transfers_with_val.groupby(["player_id", "season"], as_index=False).agg(
        num_transfers=("transfer_date", "count"),
        avg_transfer_fee=("transfer_fee", "mean"),
        last_transfer_fee=("transfer_fee", "last"),
        avg_transfer_premium=("transfer_premium", "mean"),
        last_transfer_premium=("transfer_premium", "last"),
        last_transfer_club_to=("to_club_name", "last"),
        last_transfer_club_from=("from_club_name", "last"),
    )

    # ---- Aggregate Appearances ----
    apps_agg = appearances.groupby(["player_id", "season"], as_index=False).agg(
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_minutes_played=("minutes_played", "sum"),
        total_yellow_cards=("yellow_cards", "sum"),
        total_red_cards=("red_cards", "sum"),
        matches_played=("game_id", "count")
    )

    # Merge player_name (since aggregation used IDs)
    val_agg = val_agg.merge(players[["player_id", "player_name"]], on="player_id", how="left")
    trans_agg = trans_agg.merge(players[["player_id", "player_name"]], on="player_id", how="left")
    apps_agg = apps_agg.merge(players[["player_id", "player_name"]], on="player_id", how="left")

    return players, val_agg, trans_agg, apps_agg


def merge_data(players, val_agg, trans_agg, apps_agg):
    """Merge all preprocessed datasets on player_name + season"""

    merged = val_agg.merge(trans_agg, on=["player_id", "season", "player_name"], how="outer")
    merged = merged.merge(apps_agg, on=["player_id", "season", "player_name"], how="outer")

    # Add static player info
    player_info = players[[
        "player_id", "player_name", "last_season", "current_club_id", "country_of_birth",
        "country_of_citizenship", "date_of_birth", "sub_position", "position", "foot",
        "height_in_cm", "contract_expiration_date", "agent_name", "current_club_name",
        "highest_market_value_in_eur"
    ]].drop_duplicates()

    final_df = merged.merge(player_info, on=["player_id", "player_name"], how="left")

    # Reorder
    final_df = final_df.sort_values(by=["player_name", "season"])

    return final_df


def main():
    players, valuations, transfers, appearances = load_data()
    players, val_agg, trans_agg, apps_agg = preprocess(players, valuations, transfers, appearances)
    final_df = merge_data(players, val_agg, trans_agg, apps_agg)

    out_path = r"C:\Users\asing\PycharmProjects\-Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data-\processed_data\transfermarkt_player_data.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")
    print(final_df.head())


if __name__ == "__main__":
    main()