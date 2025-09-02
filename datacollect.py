import pandas as pd
import numpy as np

def solve_problem():
    try:
        market_df = pd.read_csv('Market value data.csv')
        injury_df = pd.read_csv('Injury data.csv')
        sentiment_df = pd.read_csv('Sentiment analysis.csv')

        merged_data = clean_and_merge_data(market_df, injury_df, sentiment_df)

        if merged_data is None:
            print("Failed to merge data. Check your column names or data types.")
            return

        analyze_and_present_results(merged_data)

    except FileNotFoundError as e:
        print(f"Error: One of the files was not found. Please check the file names and paths. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def clean_and_merge_data(market_df, injury_df, sentiment_df):
    sentiment_map = {' Positive ': 1, ' Neutral ': 0, ' Negative ': -1, 'Positive': 1, 'Neutral': 0, 'Negative': -1}
    sentiment_df['Sentiment_Score'] = sentiment_df['Sentiment'].map(sentiment_map)
    
    market_df['Fee'] = pd.to_numeric(market_df['Fee'], errors='coerce')

    user_sentiment = sentiment_df.groupby('User')['Sentiment_Score'].mean().reset_index()

    user_injuries = injury_df.groupby('p_id2')['total_days_injured'].sum().reset_index()

    merged_one = pd.merge(market_df, user_injuries, left_on='Name', right_on='p_id2', how='left')

    final_merged = pd.merge(merged_one, user_sentiment, left_on='Name', right_on='User', how='left')

    final_merged['total_days_injured'] = final_merged['total_days_injured'].fillna(0)
    final_merged['Sentiment_Score'] = final_merged['Sentiment_Score'].fillna(0)

    return final_merged

def analyze_and_present_results(data):
    high_fee_threshold = data['Fee'].quantile(0.75)
    high_injury_threshold = data['total_days_injured'].quantile(0.75)

    filtered_data = data[
        (data['Fee'] >= high_fee_threshold) &
        (data['total_days_injured'] >= high_injury_threshold) &
        (data['Sentiment_Score'] < 0)
    ]

    if not filtered_data.empty:
        top_players = filtered_data.sort_values(
            by=['Fee', 'total_days_injured'],
            ascending=[False, False]
        )

        print("Top players with high fees, significant injuries, and negative public sentiment:")
        print("--------------------------------------------------------------------------")
        if len(top_players) > 5:
            top_players_to_display = top_players.head(5)
        else:
            top_players_to_display = top_players

        for _, row in top_players_to_display.iterrows():
            print(f"Name: {row['Name']}")
            print(f"  Club: {row['Club']}")
            print(f"  Transfer Fee: {row['Fee']} million")
            print(f"  Total Days Injured: {row['total_days_injured']}")
            print(f"  Average Sentiment Score: {row['Sentiment_Score']:.2f}")
            print("-" * 50)
    else:
        print("Couldn't find any players that matched all criteria.")
        print("Maybe the data doesn't support the hypothesis or the thresholds are too strict.")
        print("You might want to tweak the thresholds or the sentiment mapping.")

if __name__ == "__main__":
    solve_problem()
