import pandas as pd

def load_data():
    """
    Load the Kaggle Parkinson's dataset.
    Returns a pandas DataFrame.
    """
    # Load Kaggle dataset
    kaggle_df = pd.read_csv("parkinsons_kaggle.csv")

    # Optional: clean column names
    kaggle_df.columns = kaggle_df.columns.str.strip().str.lower()

    print(f"Kaggle dataset shape: {kaggle_df.shape}")

    return kaggle_df
