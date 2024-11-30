import pandas as pd

def load_data(filepath="data/Iris.csv"):
    return pd.read_csv(filepath)

def clean_data(data):
    # Example: Drop NA values
    return data.dropna()
