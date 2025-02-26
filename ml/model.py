import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """Train and return a machine learning model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Calculate precision, recall, and F-beta score."""
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return predictions."""
    return model.predict(X)


def save_model(model, path):
    """Save the model to a file."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Load a model from a file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model
):
    """Compute model performance on a specific data slice."""
    slice_data = data[data[column_name] == slice_value]
    X_slice, y_slice, _, _ = process_data(
        slice_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X_slice)
    return compute_model_metrics(y_slice, preds)
    