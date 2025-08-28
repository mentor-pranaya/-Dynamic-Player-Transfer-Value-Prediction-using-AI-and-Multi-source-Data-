# Data Information for TransferIQ: Dynamic Player Transfer Value Prediction

## 1. Development Environment
- **IDE Used:** PyCharm
- **Python Version:** 3.11

---

## 2. Data Sources

### 2.1 StatsBomb Football Data
- **Contributor:** Saurabh Shahane
- **Source:** Kaggle
- **Format:** JSON
- **Description:**  
  - Contains detailed football match events data including passes, shots, dribbles, tackles, and other in-game statistics.  
  - Provides player and team performance metrics at event-level granularity.  
- **Usage in Project:**  
  - Primary source for player performance statistics.  
  - Used to generate features like player efficiency, goals, assists, pass accuracy, defensive actions, etc.

---

### 2.2 Player Injuries and Team Performance Dataset
- **Contributor:** Amrit Biswas
- **Source:** Kaggle
- **Format:** CSV
- **Description:**  
  - Contains historical injury records for players along with team performance indicators.  
  - Includes columns such as `player_id`, `injury_type`, `start_date`, `end_date`, `duration_days`, `matches_missed`.  
- **Usage in Project:**  
  - Used to model the impact of injuries on player availability and performance.  
  - Generates features like `total_days_injured_last_season`, `injury_frequency`, `matches_missed`, and `injury_severity`.

---

### 2.3 Football Data from Transfermarkt
- **Contributor:** David Cariboo
- **Source:** Kaggle
- **Format:** CSV
- **Description:**  
  - Historical transfer records including player market values, transfer fees, clubs, and contract details.  
  - Includes columns like `player_id`, `valuation_date`, `valuation_value`, `transfer_fee`, `player_club_id`.  
- **Usage in Project:**  
  - Provides historical market values and actual transfer fees.  
  - Used as the primary source for training the AI model to predict transfer values.  

---

### 2.4 Social Media Sentiment
- **Source:** Reddit API  
- **Format:** CSV (after processing)  
- **Description:**  
  - Captures public sentiment regarding players through Reddit discussions and posts.  
  - Sentiment analysis performed to categorize mentions as positive, negative, or neutral.  
- **Usage in Project:**  
  - Generates sentiment-related features for each player, such as `average_sentiment_score` or `volume_of_mentions`.  
  - Adds a social factor influencing transfer market perception and potential value.  
