import tweepy
import pandas as pd
import time

# --- Authentication ---
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJoF3wEAAAAAGwIZCezor44WiDEiedMyto%2Br1js%3DazUnyouDbE5frx1vUnJ4tlRlPfMULnVZb116Hy0r0BrI9BD04Z"

# --- Main Script ---

# 1. Load the player data and select the TOP 10 most valuable players
try:
    player_df = pd.read_csv('laliga_2015_market_values.csv')
    top_players_df = player_df.sort_values(by='Market Value 2015 (in millions €)', ascending=False).head(10)
    player_names = top_players_df['Player Name'].tolist()
    print(f"✅ Loaded and selected the top {len(player_names)} most valuable players.")
    print("Top players to be searched:", player_names)

except FileNotFoundError:
    print("❌ Error: 'laliga_2015_market_values.csv' not found.")
    exit()

# 2. Authenticate with the Twitter API
try:
    # wait_on_rate_limit=True will handle the long 15-minute pauses automatically
    client = tweepy.Client(BEARER_TOKEN, wait_on_rate_limit=True)
    print("✅ Authentication with Twitter API Successful")
except Exception as e:
    print(f"❌ Error during authentication: {e}")
    exit()

# 3. Loop through the players with a retry mechanism
all_tweets_data = []
players_processed = 0
print("\n--- Starting to fetch tweets with a persistent retry logic ---")

for name in player_names:
    players_processed += 1
    print(f"--> Processing player {players_processed}/{len(player_names)}: {name}")
    
    search_query = f'"{name}" -is:retweet lang:en'
    
    # --- The Fix: A retry loop to ensure data is fetched ---
    success = False
    attempts = 0
    while not success and attempts < 3:
        try:
            tweets = client.search_recent_tweets(
                query=search_query,
                max_results=10,
                tweet_fields=["created_at", "public_metrics"]
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    all_tweets_data.append({
                        'player_name': name,
                        'created_at': tweet.created_at,
                        'text': tweet.text,
                        'like_count': tweet.public_metrics['like_count'],
                        'retweet_count': tweet.public_metrics['retweet_count']
                    })
                print(f"  -> Successfully collected {len(tweets.data)} tweets for {name}.")
            else:
                print(f"  -> No recent tweets found for {name}.")

            success = True # Mark as successful to move to the next player

        except Exception as e:
            attempts += 1
            print(f"  -> An error occurred on attempt {attempts}: {e}. Retrying after a short pause...")
            time.sleep(30) # Wait 30 seconds before retrying

    if not success:
        print(f"  -> FAILED to get data for {name} after 3 attempts.")


print("\n--- Tweet collection finished ---")

# 4. Save the collected data
if all_tweets_data:
    df_all_tweets = pd.DataFrame(all_tweets_data)
    output_filename = 'top_players_tweets.csv'
    df_all_tweets.to_csv(output_filename, index=False)
    
    print(f"\n✅ Successfully saved {len(df_all_tweets)} tweets to {output_filename}")
    print(df_all_tweets.head())
else:
    print(f"\nScript completed but found 0 relevant tweets for the top players.")