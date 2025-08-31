
#!/usr/bin/env python
import argparse, json
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger("models.baselines")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    Path("models/baselines").mkdir(parents=True, exist_ok=True)
    Path("models/baselines/median_model.json").write_text(json.dumps({"info":"median placeholder"}))
    logger.info("Saved baseline placeholder models")

if __name__ == "__main__":
    main()
