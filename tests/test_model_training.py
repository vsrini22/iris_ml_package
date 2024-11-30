import pandas as pd
from iris_ml_package.model_training import train_model, save_model
import os

def test_train_model():
    # Create a small dataset
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0]
    })

    model = train_model(data)
    assert model is not None  # Ensure the model is returned
    assert hasattr(model, "predict")  # Ensure it's a valid sklearn model

def test_save_model():
    # Mock model
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0]
    })

    model = train_model(data)
    filepath = "models/test_model.pkl"
    save_model(model, filepath)

    assert os.path.exists(filepath)  # Ensure the model file is created

    # Clean up
    os.remove(filepath)
