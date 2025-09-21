"""
premier_league_2024_squads.py
Scrape 2024/25 Premier League squad and market-value data from Transfermarkt.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# ---------------- SETTINGS ----------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com",
}
LEAGUE_NAME = "Premier League"
LEAGUE_CODE = "GB1"      # Transfermarkt code for Premier League
SEASON = 2018         # 2024/25 season
SLEEP_BETWEEN_REQUESTS = 3  # seconds
OUTPUT_FILE = f"premier_league_{SEASON}.csv"
# ------------------------------------------

def parse_value(value_str: str):
    """Convert a Transfermarkt market-value string like '€10m' to a float in EUR."""
    if not value_str or value_str == "-":
        return None
    s = value_str.replace("€", "").replace(",", "").strip().lower()
    mult = 1
    if s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("k"):
        mult, s = 1_000, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None

def scrape_squad(squad_url: str, league: str, season: int) -> pd.DataFrame:
    """Scrape one club’s squad page."""
    print(f"  Scraping squad: {squad_url}")
    resp = requests.get(squad_url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"    ❌ Failed to fetch squad page ({resp.status_code})")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "items"})
    if not table:
        print(f"    ⚠️ No squad table found at {squad_url}")
        return pd.DataFrame()

    rows = table.tbody.find_all("tr", class_=["odd", "even"])
    data = []
    for row in rows:
        name = row.find("td", class_="hauptlink")
        pos = row.find("td", class_="posrela")
        age = row.find("td", class_="zentriert")
        nationality = [img["title"] for img in row.find_all("img", class_="flaggenrahmen")]
        mv = row.find("td", class_="rechts hauptlink")
        mv_val = parse_value(mv.get_text(strip=True)) if mv else None

        data.append({
            "name": name.get_text(strip=True) if name else None,
            "position": pos.get_text(strip=True) if pos else None,
            "age": age.get_text(strip=True) if age else None,
            "nationality": nationality,
            "market_value_raw": mv.get_text(strip=True) if mv else None,
            "market_value_eur": mv_val,
            "club_url": squad_url,
            "league": league,
            "season": season,
        })
    return pd.DataFrame(data)

def get_club_urls(league_url: str):
    """Collect all club squad-page URLs for the league season."""
    print(f"Fetching league page: {league_url}")
    resp = requests.get(league_url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"❌ Failed to fetch league page (status {resp.status_code})")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "items"})
    if not table:
        print("⚠️ No clubs table found on league page.")
        return []

    clubs = table.find_all("td", class_="hauptlink no-border-links")
    urls = [
        "https://www.transfermarkt.com" + c.find("a")["href"]
        for c in clubs
        if c.find("a") and "startseite" in c.find("a")["href"]
    ]
    print(f"Found {len(urls)} club URLs.")
    return urls

def main():
    league_url = (
        f"https://www.transfermarkt.com/"
        f"{LEAGUE_NAME.replace(' ', '-').lower()}/startseite/"
        f"wettbewerb/{LEAGUE_CODE}/plus/?saison_id={SEASON}"
    )
    club_urls = get_club_urls(league_url)

    all_data = []
    for url in club_urls:
        df = scrape_squad(url, LEAGUE_NAME, SEASON)
        if not df.empty:
            all_data.append(df)
        time.sleep(SLEEP_BETWEEN_REQUESTS)  # be polite to the server

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(df_all)} player records to {OUTPUT_FILE}")
    else:
        print("🛑 No data scraped. Check if the 2024/25 pages are published.")

if __name__ == "__main__":
    main()
