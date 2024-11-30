from iris_ml_package.feature_engineering import scale_features
import pandas as pd

def test_scale_features():
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 1]
    })
    scaled_data = scale_features(data)
    assert not scaled_data.empty
