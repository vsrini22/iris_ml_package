import pandas as pd
from iris_ml_package.model_training import train_model, save_model
from iris_ml_package.model_inference import load_model, make_prediction
import os

def test_load_model():
    # Train and save a model
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0]
    })

    model = train_model(data)
    filepath = "models/test_model.pkl"
    save_model(model, filepath)

    loaded_model = load_model(filepath)
    assert loaded_model is not None  # Ensure model is loaded
    assert hasattr(loaded_model, "predict")  # Ensure it's a valid model

    # Clean up
    os.remove(filepath)

def test_make_prediction():
    # Create a small dataset
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0]
    })

    model = train_model(data)
    X_test = data.drop("target", axis=1)
    predictions = make_prediction(model, X_test)

    assert len(predictions) == len(X_test)  # Ensure predictions are returned for all samples
    assert all(isinstance(p, int) for p in predictions)  # Ensure predictions are integers
