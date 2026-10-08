import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("online_shopping.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("online_shopping_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):

        model = joblib.load(
            "online_shopping_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 30,
            "Gender": "Female",
            "MonthlyIncome": 75000,
            "WebsiteVisits": 10,
            "PagesViewed": 8,
            "TimeSpentMinutes": 40,
            "PreviousPurchases": 5,
            "CartItems": 3,
            "DiscountUsed": "Yes",
            "DeviceType": "Mobile",
            "Membership": "Premium",
            "CustomerRating": 4.2,
            "PurchaseAmount": 5000
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()