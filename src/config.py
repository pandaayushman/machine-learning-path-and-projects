import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data/raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data/processed")
MODELS_DIR = os.path.join(BASE_DIR, "models/trained_models")
RESULTS_DIR = os.path.join(BASE_DIR, "models/results")

# Experiment configs
RANDOM_STATE = 42
TARGET = "progression_score"   # Column in dataset for disease progression