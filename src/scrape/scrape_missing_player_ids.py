import time
import requests
from bs4 import BeautifulSoup
import mysql.connector
import logging
import re

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    filename="tm_playerid_search.log",
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/87.0.4280.88 Safari/537.36"
}

# ---------------------------
# Function to search player
# ---------------------------
def get_transfermarkt_id(player_name):
    """Search Transfermarkt for a player name and return ID (int) if found"""
    search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name.replace(' ', '+')}"
    print(search_url)
    r = requests.get(search_url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        logging.warning(f"Search failed for {player_name}, status={r.status_code}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Try to find first player link in results
    link = soup.select_one("table.items td.hauptlink a")
    if not link:
        return None
    print(link)
    href = link["href"]  # e.g. "/lionel-messi/profil/spieler/28003"
    match = re.search(r"/spieler/(\d+)", href)
    if match:
        return int(match.group(1))
    return None

# ---------------------------
# Main Loop
# ---------------------------
def main():
    cursor.execute("SELECT id, name FROM players_trfrmrkt WHERE transfermarkt_id IS NULL")
    players = cursor.fetchall()

    for pid, name in players:
        try:
            tm_id = get_transfermarkt_id(name)
            if tm_id:
                cursor.execute(
                    "UPDATE players_trfrmrkt SET transfermarkt_id=%s WHERE id=%s",
                    (tm_id, pid)
                )
                db.commit()
                logging.info(f"✅ Updated {name} (id={pid}) with tm_id={tm_id}")
            else:
                logging.warning(f"❌ No match found for {name}")
        except Exception as e:
            logging.error(f"Error processing {name}: {e}")

        time.sleep(2)  # polite delay

if __name__ == "__main__":
    main()
    cursor.close()
    db.close()

