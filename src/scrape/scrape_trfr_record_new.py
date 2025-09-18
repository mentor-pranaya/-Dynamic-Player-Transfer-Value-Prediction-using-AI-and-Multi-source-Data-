import time
import mysql.connector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re


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
# Selenium Setup
# ---------------------------
options = webdriver.ChromeOptions()

#options.add_argument("--headless=new") 
# had to click on accept button manually once, so removing headless for now
#options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=options)
def clean_fee(fee_text):
    """Cleans and converts transfer fee text to a numeric value or reason."""
    fee = 0
    reason = "-"
    if re.search(r"[0-9]", fee_text):  # if contains digits, treat as money
        fee = fee_text  
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
        reason = fee_text

    #cleaned = fee_text.replace("€", "").replace("m", "000000").replace("k", "000").replace(",", "").strip()
    #print(f"Cleaned fee: {fee}, reason: {reason}")
    return fee, reason
def scrape_transfer_history(player_id, transfermarkt_id):
    url = f"https://www.transfermarkt.com/spieler/transfers/spieler/{transfermarkt_id}"
    driver.get(url)
    print(url)
    # Wait for and click consent banner
    """
    clicking the consent button once manually, so commenting this out for now
    had to reactivate because of site blocking access otherwise"""
    try:
        consent_button = WebDriverWait(driver, 11).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.accept-all.sp_choice_type_11"))
        )
        consent_button.click()
        print("Clicked consent.")
    except Exception as e:
        print("Consent already accepted or not found:", e)
    
    # Wait for transfer history grid with retry
    #updated scraping script to scroll and retry on not finding the transfer grid, as the code sometimes fails before the grid loads

    #Also found that the lazy load script would not let the grid load at times, till the page is scrolled till the grid, added scroll
    grid_found = False
    for attempt in range(5):  # try up to 5 times (the script was failing sometimes, before the grid appeared)
        try:
            driver.execute_script(f"window.scrollTo(0, {(attempt+1)*800});")
            time.sleep(2)  # let JS load content
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.tm-player-transfer-history-grid"))
            )
            grid_found = True
            print(f"Transfer grid loaded on attempt {attempt+1}")
            break
        except Exception as e:
            print(f"Attempt {attempt+1}: Transfer grid not found yet. Retrying...")
            time.sleep(3)  # small wait before retry
            # driver.refresh()

    if not grid_found:
        print("Transfer grid not found after 5 attempts. Skipping.")
        return


    # Find all rows (they’re in div.tm-player-transfer-history-grid > div.grid)
    rows = driver.find_elements(By.CSS_SELECTOR, "div.tm-player-transfer-history-grid")
    print(f"Found {len(rows)} transfers for player {player_id}")

    for row in rows:
        cells = row.find_elements(By.CSS_SELECTOR, "div.grid__cell")
        if not cells:
            continue

        try:
            season = cells[0].text.strip()
            #transfer_date = cells[1].text.strip()
            transfer_date = pd.to_datetime(cells[1].text.strip(), errors="coerce")
            club_left = cells[2].text.strip()
            club_joined = cells[3].text.strip()  # index may vary depending on layout
            mv = cells[4].text.strip()
            amt_mv,txt_mv=clean_fee(mv)
            fee_text = cells[5].text.strip()
            amt_fee,txt_fee=clean_fee(fee_text)
            fee, reason = amt_fee, txt_fee
            """
            # Handle fee / reason split
            fee = 0.0
            reason = "-"
            if any(char.isdigit() for char in fee_text):
                # remove currency symbols, commas, etc.
                cleaned = fee_text.replace("€", "").replace("m", "000000").replace("k", "000").replace(",", "").strip()
                try:
                    fee = float(cleaned)
                except:
                    fee = 0.0
            else:
                reason = fee_text
            """
            # insert into DB
            cursor.execute("""
                INSERT INTO player_transfer_history
                (player_id, transfermarkt_id, season, transfer_date, club_left, club_joined, mv_raw, market_value, fee_raw, fee, reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (player_id, transfermarkt_id, season, transfer_date, club_left, club_joined, mv, amt_mv,fee_text, fee, reason))
            print(f"-->Inserted transfer: {season}, Date: {transfer_date}, {club_left} -> {club_joined}, MV: {mv} / {amt_mv}, Fee: {fee_text} / {fee}, Reason: {reason}")
        except Exception as e:
            print("Error parsing row:", e)

    db.commit()

def main():
    # Example: fetch all players needing transfers
    cursor.execute("SELECT id, transfermarkt_id FROM players_trfrmrkt WHERE transfermarkt_id not in (select distinct transfermarkt_id from player_transfer_history ) and transfermarkt_id IS NOT NULL order by transfermarkt_id")
    players = cursor.fetchall()
    cur_player_count = 1
    for pid, tm_id in players:
        scrape_transfer_history(pid, tm_id)
        print(f"Processed player {cur_player_count}/{len(players)}, {len(players)-cur_player_count} Remaining (ID: {pid})")
        cur_player_count += 1
        time.sleep(2)  # polite delay

if __name__ == "__main__":
    main()
    driver.quit()
    cursor.close()
    db.close()
