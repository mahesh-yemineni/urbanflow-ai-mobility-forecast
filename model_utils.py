import joblib
import pandas as pd

MODEL_PATH = "models/urbanflow_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def preprocess(df: pd.DataFrame):
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = pd.to_datetime(df['hour'])
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['month'] = df['hour'].dt.month
    df['year'] = df['hour'].dt.year
    df['day'] = df['hour'].dt.day
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df
