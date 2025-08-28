import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

def get_injury_data(player_name):
    # Search for player on Transfermarkt
    search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        search_resp = requests.get(search_url, headers=headers)
        search_soup = BeautifulSoup(search_resp.text, "html.parser")
        player_tag = search_soup.find("a", class_="spielprofil_tooltip")
        if not player_tag:
            return None, None, None
        profile_url = "https://www.transfermarkt.com" + player_tag.get('href')
        profile_resp = requests.get(profile_url, headers=headers)
        profile_soup = BeautifulSoup(profile_resp.text, "html.parser")
        # Injury info usually in a table with the heading "Injury history"
        injury_table = profile_soup.find("table", class_="items")
        if not injury_table:
            return None, None, None
        rows = injury_table.find_all("tr")[1:]  # skip header
        # Get latest injury entry
        if len(rows) > 0:
            latest = rows.find_all("td")
            injury_type = latest[5].text.strip()
            from_date = latest.text.strip()
            until_date = latest[1].text.strip()
            return injury_type, from_date, until_date
    except Exception as ex:
        return None, None, None
    return None, None, None

# Load your CSV file
df = pd.read_csv(r'D:\Pythonproject\epl_players_basic_info_cleaned.csv')

results = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Scraping Player Injury Data"):
    player_name = row['player_name']
    injury_type, from_date, until_date = get_injury_data(player_name)
    results.append({
        "club_name": row['club_name'],
        "player_name": player_name,
        "position": row['position'],
        "injury_type": injury_type if injury_type else "None",
        "injury_start": from_date if from_date else "None",
        "injury_end": until_date if until_date else "None"
    })
    time.sleep(1)

injury_df = pd.DataFrame(results)
injury_df.to_csv("epl_players_injury_data.csv", index=False)
print("Done! Check epl_players_injury_data.csv for results.")
