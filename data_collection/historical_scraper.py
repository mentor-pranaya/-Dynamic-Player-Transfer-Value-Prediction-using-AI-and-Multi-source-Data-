import requests
import pandas as pd
import matplotlib.pyplot as plt

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

player_id = "342229" 
player_name = "Kylian Mbappé"

url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print(f"✅ Successfully fetched historical data for {player_name}")
    data = response.json()
    
    history = data.get('list', [])
    
    if history:
        df_history = pd.DataFrame(history)
        
        df_history = df_history[['datum_mw', 'y', 'verein']]
        
        df_history.rename(columns={'datum_mw': 'date', 'y': 'market_value_eur', 'verein': 'club'}, inplace=True)
        
        print(f"\nMarket value history for {player_name}:")
        print(df_history.head())
        
        # --- FIX FOR THE WARNING ---
        # We explicitly tell pandas the date format is Day.Month.Year
        df_history['date'] = pd.to_datetime(df_history['date'], format='%d.%m.%Y')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_history['date'], df_history['market_value_eur'])
        plt.title(f"Market Value History for {player_name}")
        plt.xlabel("Date")
        plt.ylabel("Market Value (EUR)")
        plt.grid(True)
        plt.show()

else:
    print(f"❌ Failed to fetch data. Status code: {response.status_code}")

    # Add this line at the end of the script to save the data
df_history.to_csv('transfermarkt_history.csv', index=False)
print("\n✅ Data saved to transfermarkt_history.csv")