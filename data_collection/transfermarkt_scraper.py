import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

player_name = "lionel-messi"
player_id = "28003"
url = f"https://www.transfermarkt.com/{player_name}/profil/spieler/{player_id}"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print(f"✅ Successfully fetched the page for {player_name.replace('-', ' ').title()}")
    
    soup = BeautifulSoup(response.content, 'lxml')
    
    try:
        market_value_element = soup.find('a', class_='data-header__market-value-wrapper')
        
        if market_value_element:
            full_text = market_value_element.text.strip()
            # Split the string at "Last update" and take the first part
            market_value = full_text.split('Last update:')[0].strip()
            
            print(f"💰 Current Market Value: {market_value}")
        else:
            print("❌ Could not find the main market value element on the page.")

    except Exception as e:
        print(f"An error occurred while parsing the page: {e}")

else:
    print(f"❌ Failed to fetch the page. Status code: {response.status_code}")