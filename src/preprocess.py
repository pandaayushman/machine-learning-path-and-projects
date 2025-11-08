import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess combined Parkinson's dataset.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Drop missing values (if any)
    df = df.dropna()

    # Assume 'status' is target (0=healthy, 1=Parkinson’s)
    if 'status' not in df.columns:
        raise ValueError("Target column 'status' not found in dataset")

    X = df.drop(columns=['status'])
    y = df['status']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
