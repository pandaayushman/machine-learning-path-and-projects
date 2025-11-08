import os
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model_trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import visualize_results

# Paths
DATA_PATH = "data/parkinsons_dataset.csv"
MODEL_DIR = "models/trained_models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        return
    
    # Preprocess
    X, y, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    preds, mae, r2 = evaluate_model(model, X_test, y_test)
    
    # Save model and scaler
    joblib.dump(model, f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    
    # Visualize and save results
    visualize_results(df, preds, y_test)
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
