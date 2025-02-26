import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None
):
    """Process the data used in the machine learning pipeline."""
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            y = np.array([])

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(pred):
    """Convert numeric prediction to string label."""
    return ">50K" if pred[0] == 1 else "<=50K"