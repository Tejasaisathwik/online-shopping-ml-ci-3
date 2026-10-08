from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_PATH = "online_shopping_purchase_model.pkl"

FEATURES = [
    "Age",
    "Gender",
    "MonthlyIncome",
    "WebsiteVisits",
    "PagesViewed",
    "TimeSpentMinutes",
    "PreviousPurchases",
    "CartItems",
    "DiscountUsed",
    "DeviceType",
    "Membership",
    "CustomerRating",
    "PurchaseAmount"
]


def load_model():
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "online-shopping-purchase-prediction"
    })


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()

    prediction_code = int(model.predict(sample)[0])

    prediction = (
        "PURCHASED"
        if prediction_code == 1
        else "NOT PURCHASED"
    )

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
