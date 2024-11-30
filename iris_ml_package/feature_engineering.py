import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(data):
    """Scale the features of the dataset."""
    try:
        features = data.drop("variety", axis=1)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
        scaled_df["variety"] = data["variety"].values
        return scaled_df
    except KeyError as e:
        raise KeyError(f"Error scaling features: {e}")
