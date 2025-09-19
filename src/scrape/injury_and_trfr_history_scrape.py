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
    user="root",
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
    # Changed to multipage scraping for injuries.
    page = 1
    season_line1 = ''
    curseason=''
    repeated_page=0
    injuries = []
    while True:
        url = f"https://www.transfermarkt.com/spieler/verletzungen/spieler/{transfermarkt_id}/page/{page}"
        print(url)
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            logging.warning(f"Failed to fetch {url} (status {r.status_code})")
            return []

        #Bug: the following code only returns first 6 rows, despite of the table having 15 rows
        # Resolved
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"class": "items"})

        if not table:
            return []

        #rows = table.find_all("tr", {"class": ["odd", "even"]})
        rows = [r for r in table.find_all("tr") if r.find_all("td")]
        #print(rows)
        row_cnt=0
        for row in rows:
            row_cnt=row_cnt+1
            print(len(rows))
            '''
            compnamecol=row.find("img",{"class":"tiny_wappen"})
            compname=compnamecol.get("alt")
            print(compname)
            #cols = [c.get_text(strip=True) for c in row.find_all("td")]
            #if len(cols) < 6:
            #    continue '''
            cols = row.find_all("td")
            curseason=cols[0].get_text(strip=True)
            if row_cnt==1:
                if season_line1==curseason:
                    print("repeat")
                    repeated_page=1
                    break
                else:
                    season_line1=curseason
            injury_name = cols[1].get_text(strip=True)
            print(cols[0].get_text(strip=True),cols[1].get_text(strip=True),cols[2].get_text(strip=True),cols[3].get_text(strip=True),cols[4].get_text(strip=True),cols[5].get_text(strip=True))
            start_date = pd.to_datetime(cols[2].get_text(strip=True), dayfirst=True, errors="coerce")
            try:
                end_date = pd.to_datetime(cols[3].get_text(strip=True), dayfirst=True)
            except:
                end_date = '1990-01-01'  # placeholder for ongoing injuries
            print(injury_name, start_date, end_date)
            if pd.isna(end_date):
                end_date = '1990-01-01'  # placeholder for ongoing injuries
            print(injury_name, start_date, end_date)
             # Days out and games missed
            days_out = None
            games_missed = None
            
            duration_txt = cols[4].get_text(strip=True)
            try:
                days_out = int(duration_txt.lower().replace("days", "").strip())
            except:
                days_out = None
            
            # Last cell: club(s) + span
            last_td = cols[5]

            # get all clubs in that cell
            clubs = [img["alt"] for img in last_td.find_all("img", alt=True)]
            clubs_text = ", ".join(clubs)

            # games missed is the <span> at the end
            span = last_td.find("span")
            try:
                games_missed = int(span.get_text(strip=True))
            except:
                games_missed = None


            #competition = cols[5]
            print((injury_name, start_date, end_date, days_out, games_missed, clubs_text))
            injuries.append((injury_name, start_date, end_date, days_out, games_missed, clubs_text))
        
            #print(injuries)
            print(f"Got {len(injuries)} rows")
            
        page += 1
        time.sleep(1)
        if repeated_page:
            break
    return injuries

def save_injuries(player_id, transfermarkt_id, injuries):
    for inj in injuries:
        print(inj)
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
    print(url)
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        logging.warning(f"Failed to fetch transfer history {url} (status {r.status_code})")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    
    #table = soup.find("table", {"class": "items"})
    grids = soup.find("div", {"class":"grid tm-player-transfer-history-grid"})
    print(f"grids {grids}")
    if not table:
        return []

    transfers = []
    '''rows = table.find_all("tr", {"class": ["odd", "even"]})
    print(rows)'''
    for row in grids:
        '''cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) < 6:
            continue '''
        print(row)
        season = row.select_one(".tm-player-transfer-history-grid__season")
        transfer_date = row.select_one(".tm-player-transfer-history-grid__date")
        club_left = row.select_one(".tm-player-transfer-history-grid__old-club a.tm-player-transfer-history-grid__club-link")
        market_value = row.select_one(".tm-player-transfer-history-grid__new-club a.tm-player-transfer-history-grid__club-link")
        market_value = row.select_one(".tm-player-transfer-history-grid__market-value")
        fee_raw = row.select_one(".tm-player-transfer-history-grid__fee")
        
        '''season = cols[0]
        transfer_date = pd.to_datetime(cols[1], dayfirst=True, errors="coerce")
        club_left = cols[2]
        club_joined = cols[3]
        market_value = cols[4]
        fee_raw = cols[5]'''

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
        print(season, transfer_date, club_left, club_joined, market_value, fee, reason)
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
    cursor.execute("SELECT id, transfermarkt_id FROM players_trfrmrkt WHERE transfermarkt_id IS NOT NULL order by transfermarkt_id")
    #cursor.execute("SELECT id, transfermarkt_id FROM players_trfrmrkt WHERE transfermarkt_id =72522")
    players = cursor.fetchall()

    done = set()
    if os.path.exists("injuries_done.txt"):
        with open("injuries_done.txt", "r") as f:
            done = set(line.strip() for line in f)
    playercnt=1
    for pid, tm_id in players:
        if str(tm_id) in done:
            continue

        try:
            # Injuries
            injuries = scrape_injuries(tm_id)
            print(injuries)
            if injuries:
                save_injuries(pid, tm_id, injuries)
                logging.info(f"Saved {len(injuries)} injuries for player_id={pid} (tm_id={tm_id})")
                print(f"Injuries saved for Player #{playercnt}") 
            # Transfers
            '''
            transfers = scrape_transfer_history(tm_id)
            print(transfers)
            if transfers:
                save_transfer_history(pid, tm_id, transfers)
                logging.info(f"Saved {len(transfers)} transfers for player_id={pid} (tm_id={tm_id})")
                print(f"Transfers saved for Player #{playercnt}")
            '''
        except Exception as e:
            logging.error(f"Error scraping player_id={pid} tm_id={tm_id}: {e}")
        # mark as done
        with open("injuries_done.txt", "a") as f:
            f.write(str(tm_id) + "\n")

        time.sleep(2)  # polite delay
        playercnt=playercnt+1
if __name__ == "__main__":
    main()
    cursor.close()
    db.close()

