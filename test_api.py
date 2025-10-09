import requests
import json

# The URL where your Flask API is running
url = 'http://127.0.0.1:5000/predict'

# Example data for a hypothetical player.
# This dictionary MUST contain all 18 feature names from your model.
player_data = {
    "goals": 15,
    "assists": 20,
    "successful_passes": 1800,
    "tackles_won": 40,
    "avg_sentiment_score": 0.35,
    "total_days_injured": 10,
    "injury_count": 1,
    "position_Midfielder": 1,
    "position_Winger": 0,
    "Nationality_Brazilian": 0,
    "Nationality_Colobian": 0,
    "Nationality_Croatian": 1,
    "Nationality_French": 0,
    "Nationality_German": 0,
    "Nationality_Portuguese": 0,
    "Nationality_Spanish": 0,
    "Nationality_Uruguay": 0,
    "Nationality_Wales": 0
    # Note: The 'lstm_prediction' is NOT needed here, as the final model
    # should be saved without it for a real-world prediction.
    # The API will need to be updated to generate it if needed.
    # For now, we assume the saved model expects these 18 features.
}


try:
    # Send the data to the API as a POST request
    response = requests.post(url, json=player_data)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Print the prediction received from the API
    print("--- API Response ---")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")