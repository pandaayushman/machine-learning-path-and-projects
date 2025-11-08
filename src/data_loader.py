import pandas as pd

def load_data():
    """
    Load and merge the UCI and Kaggle Parkinson's datasets.
    Returns a combined pandas DataFrame.
    """
    # Load datasets
    uci_df = pd.read_csv("parkinsons_uci.csv")
    kaggle_df = pd.read_csv("parkinsons_kaggle.csv")

    # Optional: align column names if different
    kaggle_df.columns = kaggle_df.columns.str.strip().str.lower()
    uci_df.columns = uci_df.columns.str.strip().str.lower()

    # Combine datasets (only common columns)
    common_cols = list(set(uci_df.columns).intersection(set(kaggle_df.columns)))
    combined_df = pd.concat([uci_df[common_cols], kaggle_df[common_cols]], ignore_index=True)

    print(f"Loaded UCI: {uci_df.shape}, Kaggle: {kaggle_df.shape}, Combined: {combined_df.shape}")

    return combined_df
