import requests
import json
import os
from dotenv import load_dotenv
# IMPORTANT: Replace this with your actual Bearer Token
# Keep this token private and never push it to GitHub!
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# The search query. Let's search for recent tweets about Cristiano Ronaldo.
search_query = "Cristiano Ronaldo -is:retweet lang:en"
# '-is:retweet' filters out retweets, 'lang:en' gets only English tweets.

# The URL for the recent search endpoint
url = "https://api.twitter.com/2/tweets/search/recent"

# Set up the headers for authentication
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# Set up the query parameters
params = {
    'query': search_query,
    'max_results': 10  # Get up to 10 tweets
}

print("Connecting to the X API...")

# Make the GET request to the API
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    print("✅ Successfully connected and got a response!")
    # Parse the JSON response
    response_data = response.json()
    # Pretty print the JSON so it's easy to read
    print(json.dumps(response_data, indent=4, sort_keys=True))
elif response.status_code == 401:
     print("❌ Authentication Error (401): Check if your Bearer Token is correct.")
else:
    print(f"❌ An error occurred. Status code: {response.status_code}")
    print("Response:", response.text)