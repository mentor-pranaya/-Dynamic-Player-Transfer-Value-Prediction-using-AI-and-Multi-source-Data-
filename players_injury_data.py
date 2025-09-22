import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

# ------------------ Configuration ------------------
CSV_FILE = "C:/Users/ghans/OneDrive/Desktop/ai_project/fifa_players_cleaned.csv"
OUTPUT_DIR = Path("C:/Users/ghans/OneDrive/Desktop/ai_project/injury_history.csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Load Players ------------------
df_players = pd.read_csv(CSV_FILE)
players = df_players.iloc[:, 0].dropna().unique()
print(f"✅ {len(players)} players loaded from CSV")

# ------------------ Requests Session ------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.co/"
})

# ------------------ Helper Functions ------------------
def get_player_slug_and_id(player_name: str):
    """
    Search Transfermarkt for a player and return slug and ID.
    """
    query = player_name.replace(" ", "+")
    url = f"https://www.transfermarkt.co/schnellsuche/ergebnis/schnellsuche?query={query}"

    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        return None, None

    soup = BeautifulSoup(resp.text, "lxml")
    player_tag = soup.select_one("table.items td.hauptlink a")
    if not player_tag:
        return None, None

    href_parts = player_tag["href"].strip("/").split("/")
    if len(href_parts) >= 4:
        return href_parts[0], href_parts[-1]

    return None, None

def retrieve_injury_data(player_name: str):
    """
    Fetch injury history for a player from Transfermarkt.
    """
    slug, player_id = get_player_slug_and_id(player_name)
    if not slug or not player_id:
        print(f"❌ Player not found: {player_name}")
        return

    url = f"https://www.transfermarkt.co/{slug}/verletzungen/spieler/{player_id}"
    print(f"🔎 Accessing {player_name} -> {url}")

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"❌ HTTP {resp.status_code} for {player_name}")
            return

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", class_="items")
        if not table:
            print(f"⚠️ No injury records for {player_name}")
            return

        rows = table.find_all("tr")[1:]
        injuries = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 5:
                injuries.append({
                    "Injury": cells[0].get_text(strip=True),
                    "From": cells[1].get_text(strip=True),
                    "Until": cells[2].get_text(strip=True),
                    "Days Missed": cells[3].get_text(strip=True),
                    "Games Missed": cells[4].get_text(strip=True),
                })

        if injuries:
            df_injuries = pd.DataFrame(injuries)
            out_file = OUTPUT_DIR / f"{player_name.replace(' ', '_')}_injuries.csv"
            df_injuries.to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"✅ Saved injury data for {player_name}")
        else:
            print(f"⚠️ No injuries listed for {player_name}")

    except Exception as e:
        print(f"❌ Error fetching {player_name}: {e}")

    # Avoid being blocked
    time.sleep(2)

# ------------------ Main Execution ------------------
for player in players:
    retrieve_injury_data(player)
