from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_driver():
    """Sets up a headless Chrome driver."""
    options = Options()
    options.add_argument("--headless=new") # Use the new headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    return driver

def handle_consent(driver, url):
    """Navigates to a URL and robustly handles the GDPR consent button."""
    driver.get(url)
    try:
        wait = WebDriverWait(driver, 10)
        # **FIX 1**: Find iframe by title, which is more stable than a dynamic ID
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, '//iframe[@title="SP Consent Message"]')))
        accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="ACCEPT ALL"]')))
        accept_button.click()
        driver.switch_to.default_content()
        print("--> Main consent banner handled.")
    except Exception:
        print("--> Main consent banner not found or could not be handled.")

def scrape_squad_data(club_url, driver):
    """Scrapes all player names and their 2015 market values from a club's squad page."""
    squad_data = []
    squad_url = club_url.replace("/startseite/", "/kader/")
    driver.get(squad_url)
    
    try:
        wait = WebDriverWait(driver, 15)
        
        # **FIX 2**: Wait for the player table to be present in the HTML.
        # This is more reliable than waiting for it to be "visible", which can be blocked by ads.
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.items")))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        player_rows = soup.select("table.items > tbody > tr")
        
        for row in player_rows:
            player_cell = row.select_one("td.hauptlink")
            market_value_cell = row.select_one("td.rechts.hauptlink")
            
            if not player_cell or not market_value_cell:
                continue

            player_name = player_cell.get_text(strip=True)
            market_value_raw = market_value_cell.get_text(strip=True)
            
            value_in_millions = 0.0
            if 'm' in market_value_raw:
                value_in_millions = float(market_value_raw.replace('€', '').replace('m', ''))
            elif 'k' in market_value_raw:
                value_in_millions = float(market_value_raw.replace('€', '').replace('k', '')) / 1000
            
            squad_data.append({
                'Player Name': player_name,
                'Market Value 2015 (in millions €)': value_in_millions,
                'Club URL': club_url # For debugging
            })
            
    except Exception as e:
        print(f"--- Could not process club {squad_url}. Error: {e}")
        
    return squad_data

if __name__ == "__main__":
    LA_LIGA_2015_URL = "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1/plus/?saison_id=2015"
    driver = get_driver()
    
    handle_consent(driver, LA_LIGA_2015_URL)
    
    print("--- Step 1: Finding Club URLs ---")
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    club_tags = soup.select("td.hauptlink a[href*='/startseite/verein/']")
    club_urls = list({"https://www.transfermarkt.com" + tag['href'] for tag in club_tags})
    print(f"Found {len(club_urls)} clubs.")
    
    print("\n--- Step 2: Scraping Squad Data for Each Club ---")
    all_player_data = []
    for club_url in club_urls:
        # **FIX 3**: Correctly parse the club name for the print statement
        club_name = club_url.split('/startseite/')[0].split('/')[-1]
        print(f"Processing club: {club_name}")
        
        squad_data = scrape_squad_data(club_url, driver)
        if squad_data:
            all_player_data.extend(squad_data)
            print(f"  -> Found {len(squad_data)} players.")
        else:
            print(f"  -> Found 0 players.")
        time.sleep(1) # Be respectful

    print("\n--- Step 3: Saving all data to CSV ---")
    df = pd.DataFrame(all_player_data)
    df.to_csv('laliga_2015_market_values.csv', index=False)
    
    print("\n✅ Scraping complete!")
    print(f"Successfully scraped data for {len(df)} players.")
    print(df.head())
    
    driver.quit()