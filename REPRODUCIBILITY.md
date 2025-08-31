
# Reproducibility Checklist
- [x] `environment.yml` and `requirements.txt`
- [x] Global seed = 2025 (numpy, random, torch)
- [x] Deterministic dataloaders where applicable
- [x] MLflow logging with run params + metrics
- [x] Scripts use config files in `configs/`
- [x] One-liner reproduction:
```bash
conda env create -f environment.yml && conda activate transferiq
pip install -r requirements.txt
bash scripts/reproduce_full_run.sh
```
