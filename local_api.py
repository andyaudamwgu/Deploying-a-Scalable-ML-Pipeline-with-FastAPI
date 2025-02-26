import requests

# Send a GET request to http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")  # GET request

# Print the status code and welcome message
print(r.status_code)  # Print status code
print(r.json())  # Print welcome message (assuming JSON response)

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request using the data above
r = requests.post("http://127.0.0.1:8000/predict_salary/", json=data)  # POST request

# Print the status code and result
print(r.status_code)  # Print status code
print(r.json())  # Print result (assuming JSON response)