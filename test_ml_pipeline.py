import unittest
import json
import os


class TestMLPipeline(unittest.TestCase):

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_f1_score_exists(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("f1_score", metrics)

    def test_accuracy_exists(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
