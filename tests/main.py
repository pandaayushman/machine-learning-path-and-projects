import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_data, scale_features
from src.train import train_and_save_models
from src.config import TARGET

def main():
    df = load_data("parkinsons_updrs.csv")
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_scaled, scaler = scale_features(X)
    results = train_and_save_models(X_scaled, y, scaler)

    print("Training complete. Model MAE Results:", results)

if __name__ == "__main__":
    main()
