import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

# --- Config ---
CSV_PATH = "/Users/veerababu/Downloads/Complete_players_list_unique.csv"
SAVE_DIR = Path("/Users/veerababu/Downloads/injury_history")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Players ---
players_df = pd.read_csv(CSV_PATH)
players_list = players_df.iloc[:,0].dropna().unique().tolist()
print(f"✅ Loaded {len(players_list)} players from CSV")

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
    """Search Transfermarkt and return player slug + ID"""
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
        return

    url = f"https://www.transfermarkt.co/{slug}/verletzungen/spieler/{player_id}"
    print(f"🔎 Fetching {player_name} -> {url}")

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"❌ Failed: {player_name} ({resp.status_code})")
            return

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", class_="items")
        if not table:
            print(f"⚠️ No injury table found for {player_name}")
            return

        rows = table.find_all("tr")[1:]
        injuries = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) >= 5:
                injuries.append({
                    "Injury": tds[0].get_text(strip=True),
                    "From": tds[1].get_text(strip=True),
                    "Until": tds[2].get_text(strip=True),
                    "Days Missed": tds[3].get_text(strip=True),
                    "Games Missed": tds[4].get_text(strip=True),
                })

        if injuries:
            df = pd.DataFrame(injuries)
            out_path = SAVE_DIR / f"{player_name.replace(' ', '_')}_injuries.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved {player_name} injuries -> {out_path}")
        else:
            print(f"⚠️ No injuries found for {player_name}")

    except Exception as e:
        print(f"❌ Error with {player_name}: {e}")

    # polite crawling delay
    time.sleep(2)

# --- Run for all players ---
for player in players_list:
    fetch_injury_history(player)
