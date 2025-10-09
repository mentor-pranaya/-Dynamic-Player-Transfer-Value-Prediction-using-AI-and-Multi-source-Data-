import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

# --- Config ---
CSV_PATH = "/Users/veerababu/Desktop/Infosys/CompleteList.csv"
SAVE_PATH = Path("/Users/veerababu/Downloads/all_players_injuries.csv")

# --- Load Players ---
players_df = pd.read_csv(CSV_PATH)
players_list = players_df.iloc[:,0].dropna().unique().tolist()
print(f"✅ Loaded {len(players_list)} players from CSV")

# --- Load already saved injuries if exists ---
if SAVE_PATH.exists():
    existing_df = pd.read_csv(SAVE_PATH)
    processed_players = set(existing_df["Player"].unique())
    print(f"🔄 Resuming: already have {len(processed_players)} players saved")
else:
    existing_df = pd.DataFrame()
    processed_players = set()

# --- Request Session ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.co/"
})

def search_player_id(player_name):
    search_url = f"https://www.transfermarkt.co/schnellsuche/ergebnis/schnellsuche?query={player_name.replace(' ', '+')}"
    resp = session.get(search_url, timeout=30)
    if resp.status_code != 200:
        return None, None
    soup = BeautifulSoup(resp.text, "lxml")
    link = soup.select_one("table.items td.hauptlink a")
    if not link:
        return None, None
    href = link["href"]
    parts = href.strip("/").split("/")
    if len(parts) >= 4:
        slug = parts[0]
        player_id = parts[-1]
        return slug, player_id
    return None, None

def fetch_injury_history(player_name):
    slug, player_id = search_player_id(player_name)
    if not slug or not player_id:
        print(f"❌ Could not find ID for {player_name}")
        return []
    url = f"https://www.transfermarkt.co/{slug}/verletzungen/spieler/{player_id}"
    print(f"🔎 Fetching {player_name} -> {url}")
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"❌ Failed: {player_name} ({resp.status_code})")
            return []
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", class_="items")
        if not table:
            print(f"⚠️ No injury table for {player_name}")
            return []
        rows = table.find_all("tr")[1:]
        injuries = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) >= 5:
                injuries.append({
                    "Player": player_name,
                    "Injury": tds[0].get_text(strip=True),
                    "From": tds[1].get_text(strip=True),
                    "Until": tds[2].get_text(strip=True),
                    "Days Missed": tds[3].get_text(strip=True),
                    "Games Missed": tds[4].get_text(strip=True),
                })
        return injuries
    except Exception as e:
        print(f"❌ Error with {player_name}: {e}")
        return []
    finally:
        time.sleep(2)

# --- Run for remaining players ---
all_injuries = []

for player in players_list:
    if player in processed_players:
        print(f"⏩ Skipping {player} (already processed)")
        continue

    injuries = fetch_injury_history(player)
    if injuries:
        new_df = pd.DataFrame(injuries)
        if existing_df.empty:
            existing_df = new_df
        else:
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Save after each player (safe progress)
        existing_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(existing_df)} rows so far")

print(f"\n🎉 Done! Final dataset saved at: {SAVE_PATH}")
