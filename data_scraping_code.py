import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
import os

#can change values based on leagues
BASE_URL = "https://www.transfermarkt.com"
LEAGUE_SLUG = "premier-league"
LEAGUE_CODE = "GB1"
SEASON_ID = 2024
OUT_CSV = "epl_players_basic_info_24.csv"


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36",
]

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

def safe_get(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=get_headers(), timeout=15)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code == 503:
                wait = 3 * (attempt + 1)
                print(f"503 error at {url}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.RequestException:
            time.sleep(3)
            continue
    return None

def _mv_text_to_eur(txt: str):
    if not txt:
        return None
    txt = txt.strip().replace("\xa0", " ")
    if txt in {"-", "—"}:
        return None
    m = re.search(r"€\s*([\d.,]+)\s*([mk])?", txt, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    suf = (m.group(2) or "").lower()
    if suf == "m":
        num *= 1_000_000
    elif suf == "k":
        num *= 1_000
    return int(round(num))

def get_premier_league_clubs():
    url = f"{BASE_URL}/{LEAGUE_SLUG}/startseite/wettbewerb/{LEAGUE_CODE}/plus/?saison_id={SEASON_ID}"
    r = safe_get(url)
    if not r: return []
    soup = BeautifulSoup(r.text, "lxml")

    clubs = {}
    for a in soup.select("a[href*='/startseite/verein/']"):
        m = re.search(r"/verein/(\d+)", a.get("href", ""))
        if m:
            cid = int(m.group(1))
            name = a.text.strip()
            if name and (cid not in clubs or len(name) > len(clubs[cid])):
                clubs[cid] = name
    return [{"club_id": cid, "club_name": clubs[cid]} for cid in clubs]

def scrape_club_players(club_id, club_name):
    url = f"{BASE_URL}/club/startseite/verein/{club_id}/saison_id/{SEASON_ID}"
    r = safe_get(url)
    if not r: return []

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.select_one("table.items")
    if not table:
        return []

    players = []
    for tr in table.select("tbody tr"):
        # Name + ID
        link = tr.select_one("td.posrela a[href*='/profil/spieler/']")
        if not link:
            continue
        m = re.search(r"/spieler/(\d+)", link["href"])
        player_id = m.group(1) if m else None
        player_name = link.text.strip()

        # Position
        pos_cell = tr.select_one("td.posrela table.inline-table tr:nth-of-type(2) td")
        position = pos_cell.text.strip() if pos_cell else None

        # Age → first zentriert cell with digits only
        age = None
        for td in tr.select("td.zentriert"):
            txt = td.text.strip()
            if txt.isdigit():
                age = int(txt)
                break

        # Nationality (flags)
        flags = tr.select("td.zentriert img[title]")
        nationality = " | ".join(img["title"] for img in flags if img.get("title"))

        # Market value
        mv_cell = tr.select_one("td.rechts.hauptlink")
        mv_text = mv_cell.text.strip() if mv_cell else None
        mv_eur = _mv_text_to_eur(mv_text)

        players.append({
            "club_id": club_id,
            "club_name": club_name,
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "age": age,
            "nationality": nationality,
            "market_value_text": mv_text,
            "market_value_eur": mv_eur,
        })
    return players

def main():
    # Load existing data if file already exists
    if os.path.exists(OUT_CSV):
        existing_df = pd.read_csv(OUT_CSV)
        done_clubs = set(existing_df["club_id"].unique())
        all_rows = existing_df.to_dict("records")
        print(f"Resuming... already have {len(done_clubs)} clubs scraped, {len(all_rows)} players.")
    else:
        done_clubs = set()
        all_rows = []

    clubs = get_premier_league_clubs()
    print(f"Found {len(clubs)} clubs in EPL 23/24 season.")

    for c in clubs:
        if c["club_id"] in done_clubs:
            print(f"Skipping {c['club_name']} (already scraped)")
            continue

        print(f"\nScraping {c['club_name']}...")
        rows = scrape_club_players(c["club_id"], c["club_name"])
        all_rows.extend(rows)

        # Save after each club
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        time.sleep(random.uniform(2, 5))

    print(f"\nDone. Saved {len(all_rows)} players to {OUT_CSV}")

if __name__ == "__main__":
    main()
