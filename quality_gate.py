import json
import sys


MINIMUM_F1_SCORE = 0.75


print("Reading model evaluation metrics...")


with open("metrics.json", "r") as file:
    metrics = json.load(file)


f1_score = metrics["f1_score"]


print("Model F1 Score :", round(f1_score, 4))
print("Required F1 Score:", MINIMUM_F1_SCORE)


if f1_score < MINIMUM_F1_SCORE:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)


print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
