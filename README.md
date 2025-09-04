# Week 2: Data Cleaning, Feature Engineering, and Sentiment Analysis

This folder contains scripts and outputs for Week 2 tasks.

## How to run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the pipeline:

```
python week2_pipeline.py
```

Outputs will be written to `processed/` and models to `models/`.

## Outputs

- processed/dataset_clean.csv: Cleaned main dataset
- processed/dataset_features.csv: Feature-engineered dataset (trends, injury risk, contract years)
- processed/dataset_processed.csv: Scaled numeric + one-hot encoded categoricals
- processed/competitions_clean.csv: Cleaned competitions table
- processed/players_clean.csv: Cleaned players master
- processed/player_valuations_clean.csv: Cleaned valuations
- processed/player_injuries_impact_clean.csv: Cleaned injuries impact
- processed/twitter_training_clean.csv, processed/twitter_validation_clean.csv: Cleaned social data
- processed/eda_summary.txt: Summary stats of all processed datasets
- processed/sentiment_report.txt: Preliminary sentiment model metrics

## Notes

- Sentiment baseline: TF-IDF + LogisticRegression. Artifacts are in `models/`.
- Re-running the script regenerates outputs.
