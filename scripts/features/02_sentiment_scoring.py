
#!/usr/bin/env python
import argparse, pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger("features.sentiment")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    Path("data/features").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"info":["features_sentiment_placeholder"]}).to_parquet("data/features/features_aggregates.parquet")
    logger.info("Wrote data/features/features_aggregates.parquet")

if __name__ == "__main__":
    main()
