from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def open_transfermarkt(transfermarkt_id: int):
    url = f"https://www.transfermarkt.com/spieler/transfers/spieler/{transfermarkt_id}"

    options = webdriver.ChromeOptions()
    #options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)

    driver.get(url)

    # 1️⃣ wait for and click the consent button
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

    # now page is usable:
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.tm-player-transfer-history-grid"))
    )

    # here you can now scrape transfer history …
    rows = driver.find_elements(By.CSS_SELECTOR, "div.tm-player-transfer-history-grid")
    print("Found rows:", len(rows))

    driver.quit()

open_transfermarkt(72522)
