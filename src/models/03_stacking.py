
#!/usr/bin/env python
import argparse, json
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger("models.ensemble")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--quick", type=int, default=0)
    args = ap.parse_args()
    Path("models/ensemble").mkdir(parents=True, exist_ok=True)
    Path("models/ensemble/stacker.json").write_text(json.dumps({"info":"stacker placeholder"}))
    logger.info("Saved ensemble placeholder")

if __name__ == "__main__":
    main()
