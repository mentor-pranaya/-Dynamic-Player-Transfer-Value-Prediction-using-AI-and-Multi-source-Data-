
#!/usr/bin/env python
import argparse, json
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger("models.tree")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--quick", type=int, default=0)
    args = ap.parse_args()
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models/xgb_placeholder.json").write_text(json.dumps({"info":"xgb placeholder"}))
    Path("models/lgbm_placeholder.json").write_text(json.dumps({"info":"lgbm placeholder"}))
    logger.info("Saved tree model placeholders")

if __name__ == "__main__":
    main()
