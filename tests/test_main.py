import os
import pandas as pd
import joblib
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.evaluate import evaluate_model
from src.config import TARGET, MODELS_DIR

def main():
    # Load unseen test dataset
    df = load_data("new_test_dataset.csv")
    df = clean_data(df)

    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]

    # Load scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found! Run main.py to train models first.")
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate all trained models
    results = {}
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pkl") and file != "scaler.pkl":
            model_path = os.path.join(MODELS_DIR, file)
            model_name = file.replace(".pkl", "")
            results[model_name] = evaluate_model(model_path, X_test_scaled, y_test)

    print("\n Evaluation Results on Test Dataset:\n")
    for model, metrics in results.items():
        print(f"{model}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

if __name__ == "__main__":
    main()
