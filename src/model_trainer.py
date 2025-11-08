# File: src/model_trainer.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ----------------------------
# 1. Define train_model function
# ----------------------------
def train_model(X_train, y_train, X_test=None, y_test=None, save_path=None):
    """
    Trains a RandomForestRegressor model.
    Optionally evaluates on a test set and saves the model.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Evaluation:\nRMSE: {rmse:.4f}\nR² Score: {r2:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved at: {save_path}")

    return model

# ----------------------------
# 2. Paths
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
uci_path = os.path.join(project_root, "datasets", "parkinsons_uci.csv")
kaggle_path = os.path.join(project_root, "datasets", "parkinsons_kaggle.csv")
model_save_path = os.path.join(project_root, "models", "parkinson_model.pkl")
scaler_save_path = os.path.join(project_root, "models", "scaler.pkl")

# ----------------------------
# 3. Load and combine datasets
# ----------------------------
uci_df = pd.read_csv(uci_path)
kaggle_df = pd.read_csv(kaggle_path)
data = pd.concat([uci_df, kaggle_df], ignore_index=True)
print("Datasets loaded and combined successfully.")

# ----------------------------
# 4. Preprocess data
# ----------------------------
target_column = 'target'  # replace with your actual target column
X = data.drop(columns=[target_column])
y = data[target_column]

# Fill missing values
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data preprocessing completed.")

# ----------------------------
# 5. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("Train-test split done.")

# ----------------------------
# 6. Train model
# ----------------------------
model = train_model(X_train, y_train, X_test, y_test, save_path=model_save_path)

# Save the scaler
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved at: {scaler_save_path}")

# ----------------------------
# 7. Predict on a single sample
# ----------------------------
# Example input (replace with real feature values)
new_sample = np.array([[0.2, 0.1, 0.5, 0.3]])  # must match number of features
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
print(f"Prediction for new sample: {prediction[0]:.4f}")
