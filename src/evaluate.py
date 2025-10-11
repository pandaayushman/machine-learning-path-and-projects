import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    preds = model.predict(X_test)

    return {
        "R2": r2_score(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds))
    }
