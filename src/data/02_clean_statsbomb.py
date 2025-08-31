
#!/usr/bin/env python
import argparse, pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger("clean_statsbomb")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    # Placeholder cleaning step
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    # In a real step, load raw events/matches and produce per-player features
    pd.DataFrame({"info":["statsbomb_clean_placeholder"]}).to_parquet("data/processed/statsbomb_clean.parquet")
    logger.info("Wrote data/processed/statsbomb_clean.parquet")

if __name__ == "__main__":
    main()
