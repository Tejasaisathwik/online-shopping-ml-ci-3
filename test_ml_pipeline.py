import unittest
import os
import json


class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(
            os.path.exists("online_shopping_purchase_model.pkl")
        )

    def test_metrics_exists(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_quality(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(
            metrics["accuracy"],
            0.70
        )


if __name__ == "__main__":
    unittest.main()
