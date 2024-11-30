from iris_ml_package.data_processing import load_data, clean_data

def test_load_data():
    """Test loading the dataset."""
    data = load_data("data/iris.csv")
    assert data is not None

def test_clean_data():
    """Test cleaning the dataset."""
    data = load_data("data/iris.csv")
    clean = clean_data(data)
    assert clean.isnull().sum().sum() == 0
