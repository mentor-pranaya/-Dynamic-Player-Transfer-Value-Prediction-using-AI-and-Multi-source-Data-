# scripts/statsbomb_collect.py
import pandas as pd
from statsbombpy import sb

def collect_matches(competition_id=43, season_id=106):
    """
    Collects StatsBomb match data.
    Defaults: World Cup 2018 (competition_id=43, season_id=106)
    """
    print("Fetching matches from StatsBomb API...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"Retrieved {len(matches)} matches.")
    return matches

if __name__ == "__main__":
    df_matches = collect_matches()
    df_matches.to_csv("data/raw/statsbomb/matches.csv", index=False)
    print("Saved matches to data/raw/statsbomb/matches.csv")
