from iris_ml_package.model_training import load_model
from iris_ml_package.model_inference import predict
from iris_ml_package.data_processing import load_data, clean_data
from iris_ml_package.feature_engineering import scale_features

def test_model_inference():
    """Test model inference."""
    data = clean_data(load_data("data/iris.csv"))
    scaled_data = scale_features(data)
    model = load_model("models/test_model.pkl")
    predictions = predict(model, scaled_data.drop("variety", axis=1))
    assert len(predictions) > 0
