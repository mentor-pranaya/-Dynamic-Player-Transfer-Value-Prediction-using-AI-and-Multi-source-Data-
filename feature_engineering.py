"""
feature_engineering.py
----------------------
This module performs advanced feature engineering for player performance data.
It integrates performance trends, sentiment impact, physical and skill-based indices,
and age adjustment factors to create a refined dataset ready for modeling.

Steps:
1. Load the cleaned dataset.
2. Normalize and process sentiment data.
3. Create composite indices (performance, skill, physical, consistency).
4. Save the processed dataset for modeling.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ------------------------------
# Setup logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------
# Config
# ------------------------------
INPUT_PATH = Path("data/cleaned/cleaned_dataset.csv")
OUTPUT_PATH = Path("data/processed/processed_dataset.csv")

# ------------------------------
# Helper Functions
# ------------------------------
def normalize_sentiment(score):
    """Normalize sentiment score between 0 and 1."""
    if pd.isna(score):
        return 0.5
    return (score + 1) / 2  # convert -1→0 and +1→1


def compute_age_factor(age, ideal_age=27):
    """Gaussian-style decay factor centered around ideal playing age."""
    return np.exp(-((age - ideal_age) ** 2) / (2 * 5 ** 2))  # σ=5


def compute_physical_index(row):
    """Combine height, weight, strength, and stamina into a single index."""
    return (
        0.25 * row.get("height_cm", 0)
        + 0.25 * row.get("weight_kgs", 0)
        + 0.25 * row.get("strength", 0)
        + 0.25 * row.get("stamina", 0)
    ) / 100


def compute_skill_index(row):
    """Combine dribbling, ball control, vision, and passing into a single skill index."""
    return np.mean([
        row.get("dribbling", 0),
        row.get("ball_control", 0),
        row.get("vision", 0),
        row.get("short_passing", 0),
    ])


def compute_performance_index(row):
    """Weighted combination of various aspects of a player's ability."""
    sentiment_impact = row["normalized_sentiment"] * (row["overall_rating"] / 100)
    skill = row["skill_index"] / 100
    physical = row["physical_index"]
    return round(
        (0.4 * row["overall_rating"] +
         0.3 * row["potential"] +
         0.2 * (sentiment_impact * 100) +
         0.1 * (skill * 100 + physical)),
        2
    )


# ------------------------------
# Main Feature Engineering Pipeline
# ------------------------------
def main():
    logging.info("🚀 Starting feature engineering pipeline...")

    # Load dataset
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    logging.info(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

    # Handle missing data
    df.fillna({
        "Score": 0,
        "overall_rating": df["overall_rating"].mean(),
        "potential": df["potential"].mean(),
        "age": df["age"].median()
    }, inplace=True)

    # Apply feature transformations
    logging.info("📊 Applying feature transformations...")
    df["normalized_sentiment"] = df["Score"].apply(normalize_sentiment)
    df["age_factor"] = df["age"].apply(compute_age_factor)
    df["physical_index"] = df.apply(compute_physical_index, axis=1)
    df["skill_index"] = df.apply(compute_skill_index, axis=1)

    # Performance index
    df["performance_index"] = df.apply(compute_performance_index, axis=1)

    # Consistency index — how close current rating is to potential
    df["consistency_index"] = 1 - abs(df["potential"] - df["overall_rating"]) / 100

    # Age-adjusted performance
    df["age_adjusted_performance"] = df["performance_index"] * df["age_factor"]

    # Rank players by performance
    df["performance_rank"] = df["performance_index"].rank(ascending=False)

    # Save processed dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"💾 Feature-engineered dataset saved to {OUTPUT_PATH}")
    logging.info("🎯 Feature engineering pipeline completed successfully.")


if __name__ == "__main__":
    main()
