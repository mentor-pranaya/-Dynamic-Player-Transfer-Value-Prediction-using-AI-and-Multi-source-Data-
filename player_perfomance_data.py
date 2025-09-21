"""
premier_league_2024_player_stats.py
-----------------------------------
Extracts player performance data (minutes, goals, shots, assists)
for the 2024/25 Premier League season from the StatsBomb open-data set.
"""

import json
import pathlib
from collections import defaultdict
import pandas as pd

# === 1. Set the path to the StatsBomb data directory ===
# e.g. r"C:\Users\YourName\Desktop\open-data-master\data"
BASE_PATH = pathlib.Path(r"C:\Users\ghans\OneDrive\Desktop\ai_project\open-data-master\data\competitions.json")

# === 2. Find the Premier League 2024/25 competition & season IDs ===
with open(BASE_PATH / "competitions.json", encoding="utf-8") as f:
    competitions = json.load(f)

comp_id = None
season_id = None
for c in competitions:
    if c["competition_name"] == "Premier League" and "2024" in c["season_name"]:
        comp_id = c["competition_id"]
        season_id = c["season_id"]
        print("Found Premier League season:",
              c["season_name"], f"(competition_id={comp_id}, season_id={season_id})")
        break

if comp_id is None or season_id is None:
    raise ValueError("Premier League 2024/25 season not found in competitions.json")

# === 3. Load all match IDs for that season ===
matches_path = BASE_PATH / "matches" / str(comp_id) / f"{season_id}.json"
with open(matches_path, encoding="utf-8") as f:
    matches = json.load(f)

match_ids = [m["match_id"] for m in matches]
print(f"Total matches found: {len(match_ids)}")

# === 4. Aggregate player stats across all matches ===
player_stats = defaultdict(lambda: {
    "team": None,
    "minutes": 0,
    "goals": 0,
    "shots": 0,
    "assists": 0
})

for mid in match_ids:
    events_file = BASE_PATH / "events" / f"{mid}.json"
    with open(events_file, encoding="utf-8") as f:
        events = json.load(f)

    # Track playing time per player (lineups + substitutions)
    # 1. find match length
    match_length = 0
    for e in events:
        if "minute" in e and e.get("period") in (1, 2):
            match_length = max(match_length, e["minute"])

    # 2. record lineups and substitutions
    on_pitch = {}  # player_id -> minutes played
    for e in events:
        if e["type"]["name"] == "Starting XI":
            for p in e["tactics"]["lineup"]:
                pid = p["player"]["id"]
                on_pitch[pid] = 0
                player_stats[p["player"]["name"]]["team"] = e["team"]["name"]
        if e["type"]["name"] == "Substitution":
            out_id = e["player"]["id"]
            in_id = e["substitution"]["replacement"]["id"]
            out_min = e["minute"]
            if out_id in on_pitch:
                on_pitch[out_id] += out_min  # add minutes until subbed off
            on_pitch[in_id] = out_min  # starts counting from sub minute
            # track team for subbed player
            player_stats[e["substitution"]["replacement"]["name"]]["team"] = e["team"]["name"]

    # finish minutes for players still on pitch
    for pid, start_min in on_pitch.items():
        # match_length might be ~95; we treat it as full
        on_pitch[pid] = match_length - start_min if start_min > 0 else match_length

    # add to player_stats
    id_to_name = {}
    for e in events:
        if "player" in e and e["player"]:
            id_to_name[e["player"]["id"]] = e["player"]["name"]

    for pid, mins in on_pitch.items():
        name = id_to_name.get(pid)
        if name:
            player_stats[name]["minutes"] += mins

    # Goals, shots, assists
    for e in events:
        if "player" not in e or not e["player"]:
            continue
        name = e["player"]["name"]
        if e["type"]["name"] == "Shot":
            player_stats[name]["shots"] += 1
            if e["shot"]["outcome"]["name"] == "Goal":
                player_stats[name]["goals"] += 1
        if e["type"]["name"] == "Pass":
            if e.get("pass", {}).get("goal_assist") is True:
                player_stats[name]["assists"] += 1

print(f"Total players aggregated: {len(player_stats)}")

# === 5. Save to CSV ===
df = pd.DataFrame.from_dict(player_stats, orient="index").reset_index()
df.rename(columns={"index": "player"}, inplace=True)
output_file = "premier_league_2024_player_stats.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ Saved player performance data to {output_file}")
