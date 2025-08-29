import os
import pandas as pd

# ---------- Load Raw Transfermarkt Data ----------
def load_raw_data():
    """Read raw Transfermarkt CSV datasets from local storage."""
    data_dir = r"C:/Users/Abhinav/Desktop/Project/data/transfermarkt/raw"

    players_df = pd.read_csv(os.path.join(data_dir, "players.csv"))
    values_df = pd.read_csv(os.path.join(data_dir, "player_valuations.csv"))
    transfers_df = pd.read_csv(os.path.join(data_dir, "transfers.csv"))
    apps_df = pd.read_csv(os.path.join(data_dir, "appearances.csv"))

    print("✅ Loaded raw Transfermarkt datasets")
    return players_df, values_df, transfers_df, apps_df


# ---------- Preprocessing ----------
def preprocess_transfermarkt(players, values, transfers, apps):
    """Clean + aggregate valuations, transfers and appearances data."""

    # --- Normalize player names ---
    for df, col in [(players, "name"), (transfers, "player_name"), (apps, "player_name")]:
        df["player_name"] = df[col].astype(str).str.strip().str.title()

    # --- Convert dates ---
    values["valuation_date"] = pd.to_datetime(values["date"], errors="coerce", dayfirst=True)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce", dayfirst=True)
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce", dayfirst=True)

    # --- Derive season ---
    values["season"] = values["valuation_date"].dt.year
    transfers["season"] = transfers["transfer_date"].dt.year
    apps["season"] = apps["date"].dt.year

    # --- Aggregate Valuations ---
    val_summary = (
        values.groupby(["player_id", "season"])
        .agg(
            avg_value=("market_value_in_eur", "mean"),
            max_value=("market_value_in_eur", "max"),
            min_value=("market_value_in_eur", "min"),
        )
        .reset_index()
    )

    # --- Transfer Premium Calculation ---
    values_sorted = values.sort_values(["player_id", "valuation_date"])
    transfers_sorted = transfers.sort_values(["player_id", "transfer_date"])

    transfers_aligned = pd.merge_asof(
        transfers_sorted,
        values_sorted[["player_id", "valuation_date", "market_value_in_eur"]],
        by="player_id",
        left_on="transfer_date",
        right_on="valuation_date",
        direction="backward"
    )

    # Handle missing values
    transfers_aligned["ref_market_value"] = (
        transfers_aligned["market_value_in_eur_y"].fillna(transfers_aligned["market_value_in_eur_x"])
    )

    transfers_aligned["premium"] = (
        transfers_aligned["transfer_fee"] - transfers_aligned["ref_market_value"]
    )

    trans_summary = (
        transfers_aligned.groupby(["player_id", "season"])
        .agg(
            transfers_count=("transfer_date", "count"),
            mean_fee=("transfer_fee", "mean"),
            last_fee=("transfer_fee", "last"),
            mean_premium=("premium", "mean"),
            last_premium=("premium", "last"),
            last_club_to=("to_club_name", "last"),
            last_club_from=("from_club_name", "last"),
        )
        .reset_index()
    )

    # --- Appearances Summary ---
    apps_summary = (
        apps.groupby(["player_id", "season"])
        .agg(
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            minutes=("minutes_played", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            games=("game_id", "count"),
        )
        .reset_index()
    )

    # Add back player names
    val_summary = val_summary.merge(players[["player_id", "player_name"]], on="player_id", how="left")
    trans_summary = trans_summary.merge(players[["player_id", "player_name"]], on="player_id", how="left")
    apps_summary = apps_summary.merge(players[["player_id", "player_name"]], on="player_id", how="left")

    return players, val_summary, trans_summary, apps_summary


# ---------- Merge All ----------
def merge_datasets(players, val_summary, trans_summary, apps_summary):
    """Combine aggregated valuation, transfer & appearance data into one dataset."""

    merged = (
        val_summary
        .merge(trans_summary, on=["player_id", "season", "player_name"], how="outer")
        .merge(apps_summary, on=["player_id", "season", "player_name"], how="outer")
    )

    player_static = players[
        [
            "player_id", "player_name", "last_season", "current_club_id",
            "country_of_birth", "country_of_citizenship", "date_of_birth",
            "sub_position", "position", "foot", "height_in_cm",
            "contract_expiration_date", "agent_name", "current_club_name",
            "highest_market_value_in_eur"
        ]
    ].drop_duplicates()

    final = merged.merge(player_static, on=["player_id", "player_name"], how="left")
    final = final.sort_values(["player_name", "season"])

    print(f"✅ Final dataset shape: {final.shape}")
    return final


# ---------- Main ----------
def main():
    players, values, transfers, apps = load_raw_data()
    players, val_summary, trans_summary, apps_summary = preprocess_transfermarkt(players, values, transfers, apps)
    final_df = merge_datasets(players, val_summary, trans_summary, apps_summary)

    output_path = r"C:/Users/Abhinav/Desktop/Project/processed_data/transfermarkt_players_processed.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_df.to_csv(output_path, index=False)
    print(f"📁 Saved processed Transfermarkt dataset to {output_path}")
    print(final_df.head())


if __name__ == "__main__":
    main()
