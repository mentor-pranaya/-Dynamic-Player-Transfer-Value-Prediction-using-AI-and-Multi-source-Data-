import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from utils_ts import synthesize_history, synthesize_sentiment

def prepare_dataset(df, history_len=12):
    X = []
    y = []
    names = []
    for _, row in df.iterrows():
        name = row.get('name','')
        current = row.get('current_value', 0)
        highest = row.get('highest_value', current)
        hist = synthesize_history(current, highest, length=history_len, seed_str=str(name))
        # features: last k values + basic numeric metadata
        feats = hist + [row.get('age',0), row.get('appearance',0), row.get('goals',0), row.get('assists',0)]
        X.append(feats)
        # predict next-step as small growth of current
        y.append(current * 1.01)
        names.append(name)
    X = np.array(X)
    y = np.array(y)
    return X, y, names

def train_and_save(csv_path='final_data.csv', out='ts_model.joblib'):
    df = pd.read_csv(csv_path)
    X,y,_ = prepare_dataset(df)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X,y)
    joblib.dump({'model': pipeline, 'history_len': 12}, out)
    print('Saved ts model to', out)

if __name__ == '__main__':
    train_and_save()
