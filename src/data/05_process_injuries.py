
#!/usr/bin/env python
import argparse, pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger("injuries")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"info":["injury_clean_placeholder"]}).to_parquet("data/processed/injury_clean.parquet")
    logger.info("Wrote data/processed/injury_clean.parquet")

if __name__ == "__main__":
    main()
