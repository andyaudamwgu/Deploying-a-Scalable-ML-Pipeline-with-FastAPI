import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from {data_path}")
data = pd.read_csv(data_path)

# Split the data into train and test datasets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save the model and encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Model saved to {encoder_path}")

# Load the model
model = load_model(model_path)
print(f"Loading model from {model_path}")

# Run inference on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute performance on model slices
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test,
                col,
                slicevalue,
                cat_features,
                "salary",
                encoder,
                lb,
                model
            )
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
