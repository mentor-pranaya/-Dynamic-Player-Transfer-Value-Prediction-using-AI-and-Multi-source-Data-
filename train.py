import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def build_and_train(csv_path='final_data.csv', model_out='model.joblib'):
    df = pd.read_csv(csv_path)

    # Basic feature selection: use numeric columns likely predictive
    features = [
        'age','appearance','goals','assists','minutes played','days_injured',
        'games_injured','current_value','highest_value','position_encoded','winger'
    ]
    # Some datasets contain spaces in column names; ensure they exist
    features = [c for c in features if c in df.columns]
    target = 'current_value'
    if target not in df.columns:
        raise ValueError('target column current_value not found')

    X = df[features].copy()
    y = df[target].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # save model and feature list
    joblib.dump({'model': pipeline, 'features': features}, model_out)
    print('Model trained and saved to', model_out)

if __name__ == '__main__':
    build_and_train()
