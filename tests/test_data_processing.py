from iris_ml_package.data_processing import load_data, clean_data

def test_load_data():
    data = load_data("data/iris.csv")
    assert not data.empty

def test_clean_data():
    data = load_data("data/iris.csv")
    cleaned_data = clean_data(data)
    assert cleaned_data.isnull().sum().sum() == 0
