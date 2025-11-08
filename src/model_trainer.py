# File: src/model_trainer.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
datasets_folder = os.path.join(os.path.dirname(__file__), "..", "datasets")
kaggle_path = os.path.join(datasets_folder, "parkinsons_kaggle.csv")
data = pd.read_csv(kaggle_path)
print("📂 Kaggle dataset loaded successfully.")

# ----------------------------
# 2. Preprocess data
# ----------------------------
target_column = 'status'  # now we use 'status' for classification
X = data.drop(columns=[target_column, 'name'])  # drop 'name' and target
y = data[target_column]

# Fill missing values
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("⚙️  Data preprocessing completed.")

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("📊 Train-test split done.")

# ----------------------------
# 4. Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# ----------------------------
# 5. Evaluate model
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6. Save model and scaler
# ----------------------------
model_save_path = os.path.join(os.path.dirname(__file__), "..", "models", "parkinson_classifier.pkl")
scaler_save_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(model, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"Model saved at: {model_save_path}")
print(f" Scaler saved at: {scaler_save_path}")

# ----------------------------
# 7. Predict on a single sample
# ----------------------------
# Example input (replace with real feature values)
dummy_sample = np.random.rand(1, X.shape[1])
dummy_sample_df = pd.DataFrame(dummy_sample, columns=X.columns)
scaled_sample = scaler.transform(dummy_sample_df)
prediction = model.predict(scaled_sample)
probability = model.predict_proba(scaled_sample)[0][1]
print(f"🔮 Prediction for test sample: {prediction[0]} (Parkinson's probability: {probability:.4f})")
