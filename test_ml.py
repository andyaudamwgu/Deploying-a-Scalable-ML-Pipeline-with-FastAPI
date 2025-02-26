from ml.model import train_model, inference, compute_model_metrics
import numpy as np


def test_train_model():
    """Test that train_model returns a fitted model."""
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'predict'), "Model should have predict method"


def test_inference():
    """Test that inference returns predictions of correct length."""
    X_test = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = train_model(X_test, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(X_test), "Predictions length should match input"


def test_compute_metrics():
    """Test that compute_model_metrics returns valid scores."""
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "Fbeta should be between 0 and 1"