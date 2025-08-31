
#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS=ignore
export SEED=2025

# 1) Acquire minimal sample data (user to provide API keys where needed)
python src/data/01_download_and_cache.py --sample 1 --max_players 200 --seed ${SEED} || true

# 2) Process/clean
python src/data/02_clean_statsbomb.py --seed ${SEED} || true
python src/data/03_parse_transfermarkt.py --seed ${SEED} --from_cache 1 || true
python src/data/04_process_twitter.py --seed ${SEED} --limit 200 || true
python src/data/05_process_injuries.py --seed ${SEED} || true

# 3) Feature engineering
python src/features/01_performance_trends.py --seed ${SEED}
python src/features/02_sentiment_scoring.py --seed ${SEED}
python src/features/03_injury_features.py --seed ${SEED}
python src/features/04_contract_features.py --seed ${SEED}

# 4) Baselines
python src/models/00_baselines.py --seed ${SEED}
python src/models/01_tree_models.py --seed ${SEED} --quick 1

# 5) LSTM (quick epoch)
python src/models/02_lstm_model.py --seed ${SEED} --quick 1 || true

# 6) Ensemble
python src/models/03_stacking.py --seed ${SEED} --quick 1 || true
