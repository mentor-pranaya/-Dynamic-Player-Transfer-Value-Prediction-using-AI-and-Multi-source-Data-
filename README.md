# TransferIQ Prototype

This workspace contains a minimal prototype for the TransferIQ project: a data-driven player market value predictor.

Files:
- `final_data.csv` - sample dataset (provided).
- `train.py` - trains a RandomForest pipeline and saves `model.joblib`.
- `backend.py` - FastAPI app that serves `/predict`.
- `app.py` - Streamlit UI that calls the backend.
- `requirements.txt` - Python dependencies.

Quick start (Windows PowerShell):

1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train model:

```powershell
python train.py
```

3. Start backend:

```powershell
uvicorn backend:app --reload
```

4. Start frontend (in a new terminal):

```powershell
streamlit run app.py
```

The Streamlit UI will call the backend prediction endpoint.
