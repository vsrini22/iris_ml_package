import joblib

def load_model(filepath="models/iris_rf_model.pkl"):
    return joblib.load(filepath)

def make_prediction(model, data):
    return model.predict(data)
