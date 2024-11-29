# tests/test_model.py
import pytest
from model import load_data, train_model, evaluate_model
from sklearn.metrics import accuracy_score

@pytest.fixture
def data():
    # Fixture to load and split data
    X_train, X_test, y_train, y_test = load_data()
    return X_train, X_test, y_train, y_test

def test_train_model(data):
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    assert model is not None, "Model training failed"
    
def test_evaluate_model(data):
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy >= 0.6, f"Model accuracy is too low: {accuracy:.2f}"
