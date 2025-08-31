
from pathlib import Path
import json
import pandas as pd

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: str):
    p = Path(path); ensure_parent(p)
    df.to_parquet(p)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_jsonl(records, path: str):
    p = Path(path); ensure_parent(p)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
