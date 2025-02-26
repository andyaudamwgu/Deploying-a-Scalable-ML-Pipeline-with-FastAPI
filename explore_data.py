import pandas as pd
data = pd.read_csv("data/census.csv")
print(data.head())
print("\nColumn info:")
print(data.info())
print("\nSalary distribution:")
print(data["salary"].value_counts())
print("\nCategorical feature unique values:")
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]
for col in cat_features:
    print(f"{col}: {data[col].nunique()} unique values")
