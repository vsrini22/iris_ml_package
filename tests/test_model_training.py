from iris_ml_package.data_processing import load_data, clean_data
from iris_ml_package.feature_engineering import scale_features
from iris_ml_package.model_training import train_model, load_model

def test_train_model():
    """Test training the model."""
    data = clean_data(load_data("data/iris.csv"))
    scaled_data = scale_features(data)
    train_model(scaled_data, "models/test_model.pkl")

def test_load_model():
    """Test loading the model."""
    model = load_model("models/test_model.pkl")
    assert model is not None
