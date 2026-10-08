import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):

        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertEqual(
            response.get_json()["status"],
            "ok"
        )

    def test_prediction(self):

        response = self.client.post(
            "/predict",
            json={
                "Age": 35,
                "Gender": "Female",
                "MonthlyIncome": 80000,
                "WebsiteVisits": 10,
                "PagesViewed": 12,
                "TimeSpentMinutes": 40,
                "PreviousPurchases": 5,
                "CartItems": 3,
                "DiscountUsed": "Yes",
                "DeviceType": "Mobile",
                "Membership": "Premium",
                "CustomerRating": 4.2,
                "PurchaseAmount": 5000
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertIn(
            "prediction",
            response.get_json()
        )

    def test_missing_field(self):

        response = self.client.post(
            "/predict",
            json={
                "Age": 35,
                "Gender": "Female"
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
