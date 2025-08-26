# -----------------------------------------
# STEP 0: Install Required Libraries
# -----------------------------------------
# !pip install requests beautifulsoup4 pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# -----------------------------------------
# STEP 1: Define Competitions (Add More If Needed)
# -----------------------------------------
competitions = {
    "Premier League": "https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1",
    "La Liga": "https://www.transfermarkt.com/primera-division/marktwerte/wettbewerb/ES1",
    "Bundesliga": "https://www.transfermarkt.com/bundesliga/marktwerte/wettbewerb/L1",
    "Serie A": "https://www.transfermarkt.com/serie-a/marktwerte/wettbewerb/IT1",
    "Ligue 1": "https://www.transfermarkt.com/ligue-1/marktwerte/wettbewerb/FR1",
    "Eredivisie": "https://www.transfermarkt.com/eredivisie/marktwerte/wettbewerb/NL1",
    "Super Ligue": "https://www.transfermarkt.com/super-lig/marktwerte/wettbewerb/TR1",
    "Saudi Professional Ligue": "https://www.transfermarkt.com/saudi-professional-league/marktwerte/wettbewerb/SA1",
    "UEFA champions Ligue": "https://www.transfermarkt.com/uefa-champions-league/marktwerte/pokalwettbewerb/CL",
    "Europa Ligue": "https://www.transfermarkt.com/europa-league/marktwerte/pokalwettbewerb/EL",
    "UEFA Europa Conference League": "https://www.transfermarkt.com/uefa-europa-conference-league/marktwerte/pokalwettbewerb/UCOL"
}

# -----------------------------------------
# STEP 2: Set Headers (To Mimic Browser)
# -----------------------------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/87.0.4280.88 Safari/537.36"
}

# -----------------------------------------
# STEP 3: Function to Scrape All Pages for One Competition
# -----------------------------------------
def scrape_competition(base_url, competition_name):
    all_players, all_clubs, all_values, all_competitions = [], [], [], []
    page = 1
    first_row_name='' #used to compare first name of every page to avoid infinite loop
    repeated_page=0
    while True:
        # Transfermarkt pagination format (page/1, page/2, etc.)
        paginated_url = f"{base_url}/plus/1/page/{page}"
        response = requests.get(paginated_url, headers=headers)

        # Stop if page doesn't exist anymore (404 or empty table)
        if response.status_code != 200:
            print(f"Stopped scraping {competition_name} at page {page}. No more pages.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"class": "items"})

        # If no table found, we've reached the end
        if not table:
            print(f"No table found on {competition_name} page {page}. Ending scrape.")
            break

        rows = table.find_all("tr", {"class": ["odd", "even"]})
        if not rows:
            print(f"No more player rows on {competition_name} page {page}. Ending scrape.")
            break
        # Bug fix: the website keeps on showing last page upon incrementing beyond last page
        # updating code to check the new player name in existing players name to stop infinite looping
        row_cnt=0
        for row in rows:
            row_cnt=row_cnt+1
            # Player Name
            inl_tbl=row.find("table", {"class": "inline-table"})
            inl_td=inl_tbl.find("td", {"class": "hauptlink"})
            player_tag = row.find("a")
            player_name = player_tag.text.strip() if player_tag else None
            if row_cnt==1:
                if first_row_name==player_name:
                    print("repeat")
                    repeated_page=1
                    break
                else:
                    first_row_name=player_name
            print(first_row_name, player_name, repeated_page, first_row_name==player_name)
            # Market Value
            value_tag = row.find("td", {"class": "rechts hauptlink"})
            market_value = value_tag.text.strip() if value_tag else None

            # Club Name
            club_tag = row.find("td", {"class": "zentriert"}).find_next("img")
            club_name = club_tag["alt"] if club_tag else None
            
            if player_name and market_value:
                all_players.append(player_name)
                all_values.append(market_value)
                all_clubs.append(club_name)
                all_competitions.append(competition_name)

        print(f"Scraped page {page} of {competition_name} ({len(all_players)} players so far).")
        page += 1
        time.sleep(1)  # Be polite and avoid getting blocked
        if repeated_page:
            break

    return all_players, all_clubs, all_values, all_competitions

# -----------------------------------------
# STEP 4: Scrape All Competitions
# -----------------------------------------
players, clubs, values, competitions_list = [], [], [], []

for comp_name, comp_url in competitions.items():
    p, c, v, comp = scrape_competition(comp_url, comp_name)
    players.extend(p)
    clubs.extend(c)
    values.extend(v)
    competitions_list.extend(comp)

# -----------------------------------------
# STEP 5: Save Final Dataset
# -----------------------------------------
df = pd.DataFrame({
    "Player": players,
    "Club": clubs,
    "Competition": competitions_list,
    "Market Value": values
})

print("\nFinal Dataset Summary:")
print(df.head(20))
print(f"\nTotal Players Scraped Across All Competitions: {len(df)}")

df.to_csv("market_values_all_competitions.csv", index=False)
print("\nSaved to market_values_all_competitions.csv")
