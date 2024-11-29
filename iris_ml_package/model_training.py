from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    return model

def save_model(model, filepath="models/iris_rf_model.pkl"):
    joblib.dump(model, filepath)

def load_model(filepath="models/iris_rf_model.pkl"):
    return joblib.load(filepath)

if __name__ == "__main__":
    from iris_ml_package.data_processing import load_data, clean_data
    from iris_ml_package.feature_engineering import scale_features
    from iris_ml_package.model_training import train_model, save_model

    data = load_data()
    data = clean_data(data)
    data = scale_features(data)
    model = train_model(data)
    save_model(model)

