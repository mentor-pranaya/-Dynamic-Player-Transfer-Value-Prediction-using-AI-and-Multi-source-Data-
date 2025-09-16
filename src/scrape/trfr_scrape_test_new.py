from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

def scrape_transfer_history(transfermarkt_id: int):
    url = f"https://www.transfermarkt.com/spieler/transfers/spieler/{transfermarkt_id}"

    # start Chrome (headless optional)
    options = webdriver.ChromeOptions()
    #options.add_argument("--headless=new")  # remove this to see the browser
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)

    driver.get(url)

    # 1️⃣ accept consent popup
    try:
        consent_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button.accept-all.sp_choice_type_11")
            )
        )
        consent_button.click()
        print("Clicked consent.")
    except Exception as e:
        print("Consent button not found or already accepted:", e)

    # 2️⃣ wait for transfer history component to appear
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.tm-player-transfer-history-grid"))
    )

    # 3️⃣ parse rows
    rows = driver.find_elements(By.CSS_SELECTOR, "div.tm-player-transfer-history-grid")
    transfers = []
    for row in rows:
        # each row has 6 columns (Season, Date, From, To, MV, Fee)
        cols = row.find_elements(By.CSS_SELECTOR, "div.grid__cell")
        if len(cols) >= 6:
            season = cols[0].text.strip()
            date = cols[1].text.strip()
            left_club = cols[2].text.strip()
            joined_club = cols[3].text.strip()
            mv = cols[4].text.strip()
            fee_raw = cols[5].text.strip()

            # split fee into amount vs reason
            fee_amount = 0
            reason = ""
            if fee_raw.startswith("€") or fee_raw.endswith("m") or fee_raw.endswith("k"):
                fee_amount = fee_raw
            else:
                reason = fee_raw

            transfers.append({
                "Season": season,
                "Date": date,
                "Left": left_club,
                "Joined": joined_club,
                "MarketValue": mv,
                "Fee": fee_amount,
                "Reason": reason
            })

    driver.quit()

    # convert to DataFrame for convenience
    df = pd.DataFrame(transfers)
    return df

# usage example:
if __name__ == "__main__":
    df = scrape_transfer_history(72522)  # Luuk de Jong
    print(df)
#    df.to_csv("transfer_history_72522.csv", index=False)
