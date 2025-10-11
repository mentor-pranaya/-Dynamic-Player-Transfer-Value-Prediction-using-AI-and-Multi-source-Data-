import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from train_timeseries import prepare_dataset


def fast_train(csv_path='final_data.csv', out='ts_model.joblib'):
    df = pd.read_csv(csv_path)
    # sample up to 200 rows to keep training quick
    if len(df) > 200:
        df = df.sample(200, random_state=42)
    X,y,_ = prepare_dataset(df)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X,y)
    joblib.dump({'model': pipeline, 'history_len': X.shape[1]-4}, out)
    print('Saved quick ts model to', out)

if __name__ == '__main__':
    fast_train()
