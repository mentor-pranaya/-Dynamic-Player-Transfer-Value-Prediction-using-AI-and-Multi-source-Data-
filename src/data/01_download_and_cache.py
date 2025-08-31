
#!/usr/bin/env python
"""Download and cache data for StatsBomb, Transfermarkt, Twitter, Injuries.

Usage:
  python src/data/01_download_and_cache.py --sample 1 --max_players 5000 --seed 2025
"""
import argparse, os, time, json, random
from pathlib import Path
import requests
from src.utils.logger import get_logger
from src.utils.io import save_jsonl

logger = get_logger("download")

def download_statsbomb(sample: bool = True):
    # Minimal example: competitions endpoint from the public repo
    url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = Path("data/raw/statsbomb_competitions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False))
    logger.info(f"Saved {out} with {len(data)} competitions")
    return data

def cache_transfermarkt_placeholder():
    # Intentionally not scraping here; user should provide cached dumps or run scraper with throttle.
    out = Path("data/raw/transfermarkt_raw/README.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("Place cached Transfermarkt HTML/CSV dumps here. See docs.")
    logger.info("Prepared transfermarkt cache folder")

def cache_twitter_placeholder():
    out = Path("data/raw/twitter_raw.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl([{"info":"placeholder - use src/data/04_process_twitter.py"}], str(out))
    logger.info("Prepared twitter placeholder jsonl")

def cache_injury_placeholder():
    out = Path("data/raw/injury_raw.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text("player,injury,start_date,end_date,severity\n")
    logger.info("Prepared injury placeholder csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--max_players", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    random.seed(args.seed)
    download_statsbomb(sample=bool(args.sample))
    cache_transfermarkt_placeholder()
    cache_twitter_placeholder()
    cache_injury_placeholder()
    logger.info("Download & cache placeholders complete.")

if __name__ == "__main__":
    main()
