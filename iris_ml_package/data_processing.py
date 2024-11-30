import pandas as pd

def load_data(file_path="data/iris.csv"):
    """Load the Iris dataset."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")

def clean_data(data):
    """Clean and preprocess the dataset."""
    data = data.dropna()  # Remove missing values
    return data
