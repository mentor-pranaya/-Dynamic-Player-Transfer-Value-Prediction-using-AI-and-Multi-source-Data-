
# TransferIQ — Dynamic Player Transfer Value Prediction

This repository implements a multi-source, time-aware ML pipeline to predict football player transfer values.
Data sources (exactly these four): StatsBomb Open Data, Transfermarkt, Twitter API sentiment, and historical injury records.
Scope limited to **5,000 players** (configurable). Reproducibility seed = **2025**.

## Quickstart
```bash
conda env create -f environment.yml
conda activate transferiq
# sanity check
python -m pip install -r requirements.txt
```

## Project layout
See folders below; run weekly notebooks 00–07 in order.

## Reproducibility
- All scripts accept `--seed 2025`
- Experiments tracked with MLflow (local ./mlruns by default)
