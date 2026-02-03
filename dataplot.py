import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
performance = pd.read_csv("dataset.csv")   # StatsBomb-like dataset
market = pd.read_csv("player_valuations.csv")  # Transfermarkt values
injuries = pd.read_csv("player_injuries_impact.csv")  # Injury dataset
tweets_train = pd.read_csv("twitter_training.csv")  # Twitter data
tweets_valid = pd.read_csv("twitter_validation.csv")   # Twitter validation
competitions = pd.read_csv("competitions.csv")  # Competitions
players = pd.read_csv("players.csv")  # Player info

# -----------------------------
# 1. Dataset Shapes
# -----------------------------
print("Performance Data:", performance.shape)
print("Market Value Data:", market.shape)
print("Injury Data:", injuries.shape)
print("Twitter Training Data:", tweets_train.shape)
print("Twitter Validation Data:", tweets_valid.shape)
print("Competitions Data:", competitions.shape)
print("Players Data:", players.shape)

# -----------------------------
# 2. Missing Values Check
# -----------------------------
print("\nMissing Values (first 10 cols per dataset):\n")
print("Performance:\n", performance.isnull().sum().head(10))
print("Market:\n", market.isnull().sum().head(10))
print("Injuries:\n", injuries.isnull().sum().head(10))
print("Tweets Train:\n", tweets_train.isnull().sum().head(10))
print("Players:\n", players.isnull().sum().head(10))

# -----------------------------
# 3. Visualizations
# -----------------------------

# Market Value Distribution
plt.figure(figsize=(8,5))
sns.histplot(market['market_value_in_eur'], bins=50, kde=True)
plt.title("Distribution of Player Market Values (€)")
plt.xlabel("Market Value")
plt.ylabel("Count")
plt.show()

# Player Age Distribution (from Injuries dataset)
if 'Age' in injuries.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(injuries['Age'], bins=20, kde=True)
    plt.title("Distribution of Player Ages")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

# Most Common Injury Types
if 'Injury' in injuries.columns:
    plt.figure(figsize=(10,5))
    injuries['Injury'].value_counts().head(10).plot(kind='bar')
    plt.title("Top 10 Most Common Injuries")
    plt.xlabel("Injury Type")
    plt.ylabel("Frequency")
    plt.show()

# Sentiment Labels in Twitter Data
if 'Positive' in tweets_train.columns or 'Negative' in tweets_train.columns:
    plt.figure(figsize=(6,4))
    tweets_train.iloc[:,2].value_counts().plot(kind='bar')
    plt.title("Sentiment Distribution (Training Tweets)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()
