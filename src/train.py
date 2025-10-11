from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib, os
from src.models import get_models
from src.config import MODELS_DIR

def train_and_save_models(X, y, scaler=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results[name] = mae

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")

    # Save scaler too
    if scaler:
        joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    return results