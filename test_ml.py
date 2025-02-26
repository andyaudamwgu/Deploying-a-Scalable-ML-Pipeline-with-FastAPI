import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import pandas as pd

# Sample data for testing
sample_data = pd.DataFrame({
    "age": [39, 50],
    "workclass": ["State-gov", "Self-emp-not-inc"],
    "fnlgt": [77516, 83311],
    "education": ["Bachelors", "Bachelors"],
    "education-num": [13, 13],
    "marital-status": ["Never-married", "Married-civ-spouse"],
    "occupation": ["Adm-clerical", "Exec-managerial"],
    "relationship": ["Not-in-family", "Husband"],
    "race": ["White", "White"],
    "sex": ["Male", "Male"],
    "capital-gain": [2174, 0],
    "capital-loss": [0, 0],
    "hours-per-week": [40, 13],
    "native-country": ["United-States", "United-States"],
    "salary": ["<=50K", ">50K"]
})
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]


def test_train_model_type():
    """Test that train_model returns a RandomForestClassifier."""
    X_train, y_train, _, _ = process_data(
        sample_data, cat_features, "salary", True
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), (
        "Model should be a RandomForestClassifier"
    )


def test_compute_model_metrics_range():
    """Test that compute_model_metrics returns values between 0 and 1."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "Fbeta should be between 0 and 1"


def test_process_data_shape():
    """Test that process_data outputs have expected shapes."""
    train_data = sample_data.iloc[:1]  # 1 row for train
    test_data = sample_data.iloc[1:]   # 1 row for test
    X_train, y_train, encoder, lb = process_data(
        train_data, cat_features, "salary", True
    )
    X_test, y_test, _, _ = process_data(
        test_data, cat_features, "salary", False, encoder, lb
    )
    assert X_train.shape[0] == 1, "X_train should have 1 row"
    assert y_train.shape[0] == 1, "y_train should have 1 row"
    assert X_test.shape[0] == 1, "X_test should have 1 row"
    assert X_test.shape[1] == X_train.shape[1], "Feature dimensions should match"
