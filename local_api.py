import requests

# Send GET request
get_url = "http://127.0.0.1:8000/"
get_response = requests.get(get_url)
print(f"Status Code: {get_response.status_code}")
print(f"Result: {get_response.json()['message']}")

# Send POST request
post_url = "http://127.0.0.1:8000/predict/"
post_data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
post_response = requests.post(post_url, json=post_data)
print(f"Status Code: {post_response.status_code}")
print(f"Result: {post_response.json()['result']}")
# Newline at end (implicit in editor save)