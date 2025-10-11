import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df

def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    return x_scaled, scaler
