from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add permissive CORS so local frontends and browsers can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'model.joblib'
# initialize module-level placeholders so endpoints work even if startup hasn't run yet
MODEL = None
FEATURES = None
TS_MODEL = None
TS_HISTORY_LEN = 12

class PredictRequest(BaseModel):
    features: dict

@app.on_event('startup')
def load_model():
    global MODEL
    try:
        data = joblib.load(MODEL_PATH)
        MODEL = data.get('model')
        FEATURES = data.get('features')
        print('Model loaded with features:', FEATURES)
    except Exception as e:
        MODEL = None
        FEATURES = None
        print('Model not loaded:', e)

@app.get('/')
def root():
    return {'status': 'ok'}

@app.post('/predict')
def predict(req: PredictRequest):
    if MODEL is None:
        return {'error': 'model not available. Train and save model first.'}
    feat = req.features
    # ensure order
    df = pd.DataFrame([feat])
    try:
        preds = MODEL.predict(df)
        return {'predicted_value': float(preds[0])}
    except Exception as e:
        return {'error': str(e)}


# On-demand: load timeseries model when module imported if available
try:
    ts_path = 'ts_model.joblib'
    if os.path.exists(ts_path):
        data = joblib.load(ts_path)
        TS_MODEL = data.get('model')
        TS_HISTORY_LEN = data.get('history_len', TS_HISTORY_LEN)
        print('Loaded timeseries model, history len', TS_HISTORY_LEN)
    else:
        print('ts_model.joblib not found; run train_timeseries.py to create it')
except Exception as e:
    TS_MODEL = None
    print('Error loading ts model at import:', e)


class TSRequest(BaseModel):
    name: str
    current_value: float = 0
    highest_value: float = 0


@app.post('/predict_ts')
def predict_ts(req: TSRequest):
    from utils_ts import synthesize_history, synthesize_sentiment
    name = req.name or ''
    current = float(req.current_value or 0)
    highest = float(req.highest_value or current)
    history = synthesize_history(current, highest, length=TS_HISTORY_LEN, seed_str=name)
    sentiment = synthesize_sentiment(name, length=TS_HISTORY_LEN)
    # simple forecast: if ts model available, use it; else naive ramp
    forecast = []
    if TS_MODEL is not None:
        try:
            import numpy as np
            feats = np.array([history + [0,0,0,0]])
            pred = TS_MODEL.predict(feats)
            forecast = [float(pred[0])]
        except Exception:
            forecast = [history[-1] * 1.01]
    else:
        forecast = [history[-1] * 1.01]

    return {
        'history': history,
        'sentiment': sentiment,
        'forecast': forecast
    }
