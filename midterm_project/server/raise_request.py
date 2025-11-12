import json
import requests

url = "http://localhost:8080/predict_salary"

with open("sample_customer.json") as f:
    payload = json.load(f)    


response = requests.post(url, json=payload)
print(response.json())