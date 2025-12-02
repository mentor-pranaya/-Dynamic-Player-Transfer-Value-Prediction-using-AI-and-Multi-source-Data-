import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_driver():
    """Sets up a VISIBLE (headed) undetected Chrome driver."""
    options = uc.ChromeOptions()
    # We will run in headed (visible) mode to appear more human
    # options.add_argument('--headless') 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver

def handle_consent_if_present(driver, url):
    """Navigates to a URL and handles the GDPR consent button."""
    driver.get(url)
    try:
        time.sleep(3) # Wait for iframe to appear
        wait = WebDriverWait(driver, 7)
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, '//iframe[@title="SP Consent Message"]')))
        accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="ACCEPT ALL"]')))
        accept_button.click()
        driver.switch_to.default_content()
        print("--> Main consent banner handled.")
    except Exception:
        print("--> Consent banner not found or already handled.")

if __name__ == "__main__":
    PLAYER_LIST_FILENAME = 'laliga_2015_market_values.csv'
    CLUB_SQUAD_URLS = [
        "https://www.transfermarkt.com/fc-barcelona/kader/verein/131/saison_id/2015",
        "https://www.transfermarkt.com/real-madrid/kader/verein/418/saison_id/2015"
    ]

    try:
        player_df = pd.read_csv(PLAYER_LIST_FILENAME)
        top_10_players_df = player_df.sort_values(by='Market Value 2015 (in millions €)', ascending=False).head(10)
        top_10_player_names = top_10_players_df['Player Name'].tolist()
        print("✅ Successfully loaded the top 10 players.")
    except FileNotFoundError:
        print(f"❌ Error: '{PLAYER_LIST_FILENAME}' not found.")
        exit()

    driver = get_driver()
    
    print("\n--- Step 1: Building player URL address book ---")
    player_url_map = {}
    
    # Visit a page just to handle consent first
    handle_consent_if_present(driver, "https://www.transfermarkt.com")

    for squad_url in CLUB_SQUAD_URLS:
        club_name = squad_url.split('/kader/')[0].split('/')[-1]
        print(f"--> Processing club: {club_name}")
        try:
            driver.get(squad_url)
            
            # **THE FIX**: Simulate human scrolling to trigger lazy-loaded content
            print("  -> Scrolling down to load all content...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3) # Wait for content to load after scrolling

            # Wait for player links and get them directly with Selenium
            player_link_elements = WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.spielprofil_tooltip[href*='/profil/spieler/']"))
            )
            
            for element in player_link_elements:
                player_name = element.text.strip()
                player_url = element.get_attribute('href')
                if player_name and player_name not in player_url_map:
                    player_url_map[player_name] = player_url
            print(f"  -> Successfully processed {club_name}, found {len(player_link_elements)} players.")

        except Exception as e:
            print(f"  -> FAILED to process {club_name}. Skipping. Error: {e.__class__.__name__}")
            continue
        
    print(f"\nBuilt address book for {len(player_url_map)} players.")

    # (The rest of the script remains the same and will now work)
    all_injuries_data = []
    print("\n--- Step 2: Scraping injury history for top 10 players ---")
    for player_name in top_10_player_names:
        if player_name in player_url_map:
            player_url = player_url_map[player_name]
            injury_url = player_url.replace("/profil/", "/verletzungen/")
            try:
                driver.get(injury_url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.items")))
                injury_soup = BeautifulSoup(driver.page_source, 'html.parser')
                injury_table = injury_soup.select_one("table.items > tbody")
                if injury_table:
                    for row in injury_table.find_all("tr"):
                        columns = row.find_all("td")
                        if len(columns) >= 5:
                            season, injury, date_from, date_to, days_missed = [c.text.strip() for c in columns[:5]]
                            all_injuries_data.append({ 'player_name': player_name, 'season': season, 'injury': injury, 'date_from': date_from, 'date_to': date_to, 'days_missed': days_missed })
                time.sleep(1)
            except Exception as e:
                print(f"  -> Could not scrape injury data for {player_name}. Error: {e.__class__.__name__}")
        else:
            print(f"  -> Could not find URL for {player_name} in address book.")
    driver.quit()
    
    print("\n--- Step 3: Filtering injuries for the 15/16 season ---")
    if all_injuries_data:
        full_injury_df = pd.DataFrame(all_injuries_data)
        season_15_16_injuries = full_injury_df[full_injury_df['season'] == '15/16']
        if not season_15_16_injuries.empty:
            output_filename = 'top_10_players_injury_data_2015_2016.csv'
            season_15_16_injuries.to_csv(output_filename, index=False)
            print(f"\n✅ Successfully saved {len(season_15_16_injuries)} injuries from the 15/16 season to '{output_filename}'")
            print(season_15_16_injuries.head())
        else:
            print("Found injury histories, but no injuries were recorded for these players in the 15/16 season.")
    else:
        print("No injury data was collected.")