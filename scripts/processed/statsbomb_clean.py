import os
import json
import pandas as pd
from glob import glob

# === Paths ===
BASE_DIR = "/content/drive/MyDrive/player_value_prediction_project/data/raw/statsbomb/data"
OUT_PATH = "/content/drive/MyDrive/player_value_prediction_project/data/processed/statsbomb_clean.csv"

PATH_MATCHES = os.path.join(BASE_DIR, "matches")
PATH_EVENTS = os.path.join(BASE_DIR, "events")
PATH_LINEUPS = os.path.join(BASE_DIR, "lineups")
PATH_COMPETITIONS = os.path.join(BASE_DIR, "competitions.json")

# --- Load competitions ---
with open(PATH_COMPETITIONS, "r", encoding="utf-8") as f:
    competitions = pd.DataFrame(json.load(f))

# --- Load matches metadata ---
matches_data = []
for subfolder in glob(os.path.join(PATH_MATCHES, "*")):
    for file in glob(os.path.join(subfolder, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            matches_data.extend(json.load(f))

matches_df = pd.DataFrame([{
    "match_id": str(m.get("match_id")),
    "competition_id": m.get("competition", {}).get("competition_id"),
    "season": m.get("season", {}).get("season_name"),
    "match_date": m.get("match_date"),
} for m in matches_data])

matches_df = matches_df.merge(
    competitions[["competition_id", "competition_name", "country_name"]],
    on="competition_id", how="left"
)

# --- Season normalization ---
def normalize_season(s):
    if pd.isna(s): return s
    s = str(s)
    if "/" in s and len(s.split("/")[-1]) == 2:  # 2019/20 → 2019/2020
        return s.split("/")[0] + "/20" + s.split("/")[-1]
    return s
matches_df["season"] = matches_df["season"].map(normalize_season)

# --- Player lookup ---
player_lookup = {}
for file in glob(os.path.join(PATH_LINEUPS, "*.json")):
    with open(file, "r", encoding="utf-8") as f:
        lineups = json.load(f)
    for team in lineups:
        for player in team.get("lineup", []):
            player_lookup[player["player_id"]] = player.get("player_name", f"Unknown_{player['player_id']}")

# --- Safe dict getter ---
def safe_get(d, path):
    for p in path:
        if d is None or not isinstance(d, dict): return None
        d = d.get(p)
    return d

# --- Initialize output ---
if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)
first_write = True

# --- Process events incrementally ---
for file in glob(os.path.join(PATH_EVENTS, "*.json")):
    match_id = os.path.splitext(os.path.basename(file))[0]

    with open(file, "r", encoding="utf-8") as f:
        events = json.load(f)

    rows = []
    for e in events:
        player_id = safe_get(e, ["player", "id"])
        if not player_id: continue
        player_name = player_lookup.get(player_id, f"Unknown_{player_id}")
        etype = safe_get(e, ["type", "name"])
        rows.append({
            "player_name": player_name,
            "match_id": match_id,
            "event_type": etype,
            "outcome": safe_get(e, ["outcome", "name"]),
            "xG": safe_get(e, ["shot", "statsbomb_xg"]),
            "minute": e.get("minute", 0),
        })

    if not rows: 
        continue

    df_chunk = pd.DataFrame(rows)
    df_chunk = df_chunk.merge(matches_df[["match_id", "season", "competition_name", "country_name"]],
                              on="match_id", how="left")

    # Aggregate at player+season level
    agg = df_chunk.groupby(["player_name", "season", "competition_name", "country_name"]).agg(
        passes=("event_type", lambda x: (x == "Pass").sum()),
        passes_completed=("outcome", lambda x: (x == "Complete").sum()),
        carries=("event_type", lambda x: (x == "Carry").sum()),
        pressures=("event_type", lambda x: (x == "Pressure").sum()),
        duels=("event_type", lambda x: (x == "Duel").sum()),
        duels_won=("outcome", lambda x: (x == "Won").sum()),
        fouls_committed=("event_type", lambda x: (x == "Foul Committed").sum()),
        fouls_won=("event_type", lambda x: (x == "Foul Won").sum()),
        blocks=("event_type", lambda x: (x == "Block").sum()),
        interceptions=("event_type", lambda x: (x == "Interception").sum()),
        dribbles=("event_type", lambda x: (x == "Dribble").sum()),
        shots=("event_type", lambda x: (x == "Shot").sum()),
        goals=("outcome", lambda x: (x == "Goal").sum()),
        xG=("xG", "sum"),
        minutes_played=("minute", "max"),
        matches_played=("match_id", "nunique")
    ).reset_index()

    # Append to file incrementally
    agg.to_csv(OUT_PATH, mode="a", index=False, header=first_write)
    first_write = False

print(f"✅ Incremental StatsBomb features saved to {OUT_PATH}")

# --- Final aggregation pass ---
final = pd.read_csv(OUT_PATH)
final = final.groupby(["player_name", "season", "competition_name", "country_name"], as_index=False).sum()

# Derived metrics
final["pass_accuracy"] = final["passes_completed"] / final["passes"].replace(0, pd.NA)
final["duel_win_rate"] = final["duels_won"] / final["duels"].replace(0, pd.NA)

for col in ["passes","passes_completed","carries","pressures","duels",
            "fouls_committed","fouls_won","blocks","interceptions","dribbles",
            "duels_won","shots","goals"]:
    final[f"{col}_per90"] = (final[col] / final["minutes_played"].replace(0, pd.NA)) * 90

# Overwrite clean
final.to_csv(OUT_PATH, index=False)
print(f"✅ Final aggregated StatsBomb dataset saved to {OUT_PATH}")
print(final.head())
