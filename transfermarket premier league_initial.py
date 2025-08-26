# -----------------------------------------
# STEP 0: Install Required Libraries (Run once if not installed)
# -----------------------------------------
# requests: To fetch the webpage
# beautifulsoup4: To parse HTML content
# pandas: To store scraped data in tabular form
# !pip install requests beautifulsoup4 pandas --break-system-packages

# -----------------------------------------
# STEP 1: Import Libraries
# -----------------------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd

# -----------------------------------------
# STEP 2: Define URL to Scrape
# -----------------------------------------
# Example: Transfermarkt page showing market values of players in Premier League
url = "https://www.transfermarkt.com/premier-league/marktwerte/wettbewerb/GB1"

# Set headers to mimic a browser request (helps prevent blocking)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.224 Safari/537.36"
}

# -----------------------------------------
# STEP 3: Fetch Webpage Content
# -----------------------------------------
response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Successfully fetched webpage!")
else:
    print(f"Failed to fetch page, status code: {response.status_code}")

# -----------------------------------------
# STEP 4: Parse HTML with BeautifulSoup
# -----------------------------------------
soup = BeautifulSoup(response.text, "html.parser")

# -----------------------------------------
# STEP 5: Extract Player Data
# -----------------------------------------
# Transfermarkt structures players in tables with class "items"
table = soup.find("table", {"class": "items"})

players = []
values = []
clubs = []

if table:
    rows = table.find_all("tr", {"class": ["odd", "even"]})  # Player rows have these classes
    
    for row in rows:
        inl_tbl=row.find("table", {"class": "inline-table"})
        inl_td=inl_tbl.find("td", {"class": "hauptlink"})
        # Extract player name
        player_name_tag = inl_td.find("a")
        player_name = player_name_tag.text.strip() if player_name_tag else None

        # Extract market value
        value_tag = row.find("td", {"class": "rechts hauptlink"})
        market_value = value_tag.text.strip() if value_tag else None

        # Extract club name
        club_tag = row.find("td", {"class": "zentriert"}).find_next("img")
        club_name = club_tag["alt"] if club_tag else None

        if player_name and market_value:
            players.append(player_name)
            values.append(market_value)
            clubs.append(club_name)

# -----------------------------------------
# STEP 6: Store in Pandas DataFrame
# -----------------------------------------
market_data = pd.DataFrame({
    "Player": players,
    "Club": clubs,
    "Market Value": values
})

# Display first 10 rows
print("\nScraped Market Value Data:")
print(market_data.head(10))

# Save to CSV (Deliverable for Week 1)
market_data.to_csv("market_values.csv", index=False)
print("\nSaved market values to market_values.csv")

# -----------------------------------------
# STEP 7: Quick Data Exploration
# -----------------------------------------
print("\nNumber of players scraped:", len(market_data))

# Look at top 10 highest-valued players
print("\nTop 10 Players by Market Value:")
print(market_data.head(10))
