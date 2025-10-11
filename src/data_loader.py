import pandas as pd

from src.config import DATA_RAW

def load_data(filename: str) -> pd.DataFrame:
    filepath = f"{DATA_RAW}/{filename}"
    return pd.read_csv(filepath)
