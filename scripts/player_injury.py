import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
from colorama import Fore, Style, init

# --- Initialize Colorama for colored console output ---
init(autoreset=True)

# --- Config ---
CSV_PATH = Path("C:/Users/Abhinav/Downloads/Complete_players_list_unique.csv")
SAVE_DIR = Path("C:/Users/Abhinav/Downloads/injury_history")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Players from CSV ---
players_df = pd.read_csv(CSV_PATH)
players_list = players_df.iloc[:, 0].dropna().unique().tolist()
print(Fore.GREEN + Style.BRIGHT + f"[OK] Loaded {len(players_list)} players from CSV")

# --- Setup HTTP session with Windows + Chrome UA ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.co/"
})

def search_player_id(player_name):
    """Search Transfermarkt and return (slug, player_id) if available."""
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
        print(Fore.RED + f"[MISS] Could not locate player ID for {player_name}")
        return

    url = f"https://www.transfermarkt.co/{slug}/verletzungen/spieler/{player_id}"
    print(Fore.CYAN + f"[FETCH] {player_name} → {url}")

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(Fore.RED + f"[FAIL] {player_name} (HTTP {resp.status_code})")
            return

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", class_="items")
        if not table:
            print(Fore.YELLOW + f"[EMPTY] No injury data found for {player_name}")
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
            print(Fore.GREEN + Style.BRIGHT + f"[SAVED] {player_name} → {out_path}")
        else:
            print(Fore.YELLOW + f"[INFO] No recorded injuries for {player_name}")

    except Exception as e:
        print(Fore.RED + f"[ERROR] Problem with {player_name}: {e}")

    # polite delay to avoid getting blocked
    time.sleep(2)

# --- Run for all players ---
for player in players_list:
    fetch_injury_history(player)
