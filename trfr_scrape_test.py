import requests, re, time
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

def parse_money(text):
    if not text or text.strip() == "-" or text.strip() == "":
        return 0.0
    text = text.replace("€", "").replace(",", "").strip().lower()
    m = re.match(r"([\d\.]+)(m|k)?", text)
    if not m:
        return 0.0
    val, unit = m.groups()
    val = float(val)
    if unit == "m":
        val *= 1_000_000
    elif unit == "k":
        val *= 1_000
    return val

def scrape_transfer_history(transfermarkt_id):
    url = f"https://www.transfermarkt.com/spieler/transfers/spieler/72522"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    print(soup)
    # Each row of the history is a .grid.tm-player-transfer-history-grid
    rows = soup.select("div.grid.tm-player-transfer-history-grid")
    data = []
    for row in rows:
        season = row.select_one(".tm-player-transfer-history-grid__season")
        date = row.select_one(".tm-player-transfer-history-grid__date")
        club_from = row.select_one(".tm-player-transfer-history-grid__old-club a.tm-player-transfer-history-grid__club-link")
        club_to = row.select_one(".tm-player-transfer-history-grid__new-club a.tm-player-transfer-history-grid__club-link")
        mv = row.select_one(".tm-player-transfer-history-grid__market-value")
        fee_cell = row.select_one(".tm-player-transfer-history-grid__fee")

        season_txt = season.get_text(strip=True) if season else None
        date_txt = date.get_text(strip=True) if date else None
        club_from_txt = club_from.get_text(strip=True) if club_from else None
        club_to_txt = club_to.get_text(strip=True) if club_to else None
        mv_txt = mv.get_text(strip=True) if mv else None
        fee_txt = fee_cell.get_text(" ", strip=True) if fee_cell else None

        # Distinguish fee vs reason:
        fee_value = parse_money(fee_txt)
        reason = None
        if fee_value == 0 and fee_txt not in (None, "-", ""):
            reason = fee_txt  # Free transfer, End of loan, etc.

        data.append({
            "Season": season_txt,
            "Date": pd.to_datetime(date_txt, errors="coerce"),
            "Left": club_from_txt,
            "Joined": club_to_txt,
            "MV": parse_money(mv_txt),
            "Fee": fee_value,
            "Reason": reason
        })

    return pd.DataFrame(data)

# 🔹 Example
df = scrape_transfer_history(72522)
print(df)

