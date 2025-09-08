import os
import time
import requests
import mysql.connector
import pandas as pd
from bs4 import BeautifulSoup
import logging
import re

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(
    filename="injury_scraper.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------
# DB Connection
# ---------------------------
db = mysql.connector.connect(
    host="localhost",
    user="himanshu",
    password="yahoonet",
    database="AIProject"
)
cursor = db.cursor()

# ---------------------------
# Headers for Requests
# ---------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/87.0.4280.88 Safari/537.36"
}

# ---------------------------
# Scrape Injuries
# ---------------------------
def scrape_injuries(transfermarkt_id):
    url = f"https://www.transfermarkt.com/spieler/verletzungen/spieler/{transfermarkt_id}"
    print(url)
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        logging.warning(f"Failed to fetch {url} (status {r.status_code})")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": "items"})

    if not table:
        return []

    injuries = []
    rows = table.find_all("tr", {"class": ["odd", "even"]})

    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) < 6:
            continue

        injury_name = cols[0]
        start_date = pd.to_datetime(cols[1], errors="coerce")
        end_date = pd.to_datetime(cols[2], errors="coerce")
        days_out = None
        games_missed = None

        try:
            days_out = int(cols[3].replace(" Days", "").strip())
        except:
            pass

        try:
            games_missed = int(cols[4].replace("-", "0").strip())
        except:
            pass

        competition = cols[5]

        injuries.append((injury_name, start_date, end_date, days_out, games_missed, competition))

    return injuries

def save_injuries(player_id, transfermarkt_id, injuries):
    for inj in injuries:
        cursor.execute("""
            INSERT INTO player_injuries_trfrmrkt
            (player_id, transfermarkt_id, injury, start_date, end_date, days_out, games_missed, competition)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (player_id, transfermarkt_id, *inj))
    db.commit()

# ---------------------------
# Scrape Transfer History
# ---------------------------
def scrape_transfer_history(transfermarkt_id):
    url = f"https://www.transfermarkt.com/spieler/profil/spieler/{transfermarkt_id}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        logging.warning(f"Failed to fetch transfer history {url} (status {r.status_code})")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": "items"})

    if not table:
        return []

    transfers = []
    rows = table.find_all("tr", {"class": ["odd", "even"]})

    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) < 6:
            continue

        season = cols[0]
        transfer_date = pd.to_datetime(cols[1], errors="coerce")
        club_left = cols[2]
        club_joined = cols[3]
        market_value = cols[4]
        fee_raw = cols[5]

        # Parse fee vs reason
        fee = 0
        reason = None
        if re.search(r"[0-9]", fee_raw):  # if contains digits, treat as money
            fee = fee_raw
            # Convert to number (€, m, k parsing possible)
            try:
                if fee.endswith("m"):
                    fee = int(float(fee.replace("€", "").replace("m", "").strip()) * 1_000_000)
                elif fee.endswith("k"):
                    fee = int(float(fee.replace("€", "").replace("k", "").strip()) * 1_000)
                else:
                    fee = int(fee.replace("€", "").replace(",", "").strip())
            except:
                fee = 0
        else:
            reason = fee_raw

        transfers.append((season, transfer_date, club_left, club_joined, market_value, fee, reason))

    return transfers

def save_transfer_history(player_id, transfermarkt_id, transfers):
    for t in transfers:
        cursor.execute("""
            INSERT INTO player_transfer_history
            (player_id, transfermarkt_id, season, transfer_date, club_left, club_joined, market_value, fee, reason)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (player_id, transfermarkt_id, *t))
    db.commit()

# ---------------------------
# Main Loop
# ---------------------------
def main():
    cursor.execute("SELECT id, transfermarkt_id FROM players_trfrmrkt WHERE transfermarkt_id IS NOT NULL")
    players = cursor.fetchall()

    done = set()
    if os.path.exists("injuries_done.txt"):
        with open("injuries_done.txt", "r") as f:
            done = set(line.strip() for line in f)

    for pid, tm_id in players:
        if str(tm_id) in done:
            continue

        try:
            # Injuries
            injuries = scrape_injuries(tm_id)
            if injuries:
                save_injuries(pid, tm_id, injuries)
                logging.info(f"Saved {len(injuries)} injuries for player_id={pid} (tm_id={tm_id})")

            # Transfers
            transfers = scrape_transfer_history(tm_id)
            if transfers:
                save_transfer_history(pid, tm_id, transfers)
                logging.info(f"Saved {len(transfers)} transfers for player_id={pid} (tm_id={tm_id})")

        except Exception as e:
            logging.error(f"Error scraping player_id={pid} tm_id={tm_id}: {e}")

        # mark as done
        with open("injuries_done.txt", "a") as f:
            f.write(str(tm_id) + "\n")

        time.sleep(2)  # polite delay

if __name__ == "__main__":
    main()
    cursor.close()
    db.close()

