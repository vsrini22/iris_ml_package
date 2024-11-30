import pandas as pd

def predict(model, data):
    """Make predictions using the trained model."""
    predictions = model.predict(data)
    return predictions
