from iris_ml_package.data_processing import load_data, clean_data
from iris_ml_package.feature_engineering import scale_features

def test_scale_features():
    """Test scaling of features."""
    data = clean_data(load_data("data/iris.csv"))
    scaled_data = scale_features(data)
    assert "variety" in scaled_data.columns
    assert len(scaled_data) > 0
