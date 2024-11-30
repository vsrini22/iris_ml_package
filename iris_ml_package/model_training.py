import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(data, output_path="models/model.pkl"):
    """Train a Random Forest model."""
    X = data.drop("variety", axis=1)
    y = data["variety"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print("Classification Report:\n", classification_report(y_test, predictions))
    joblib.dump(model, output_path)

def load_model(filepath):
    """Load a trained model."""
    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        raise FileNotFoundError(f"Error loading model: {e}")
