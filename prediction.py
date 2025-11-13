import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

MODEL_FILE = "salary_model.pkl"

def load_model(path: str = None):
    with open(path, "rb") as f:
        model = pickle.load(f)

    return model

def predict_avg_salary(model, dv, customer_data: dict):
    customer_x = dv.transform([customer_data])
    prediction = model.predict(customer_x)
    
    return prediction[0]

@app.route("/predict_salary", methods=["POST"])
def predict_salary():
    headers = request.headers
    payload = request.json

    model_package = load_model(MODEL_FILE)
    model = model_package['model']
    dv = model_package['dict_vectorizer'] 

    predicted_salary = predict_avg_salary(model, dv, payload)

    return jsonify({"Details": f"Predicted average salary: ${predicted_salary:.2f}"}), 201

@app.route("/ping", methods=["GET", "POST"])
def ping():
    return jsonify({"Ping": "Salary prediction server is up!!"}), 200


if __name__ == "__main__":
    # gunicorn  -b 0.0.0.0:8080 --workers 1 --threads 4 --timeout 12000 app:app --reload
    app.run()