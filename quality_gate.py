import json

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)

# Minimum acceptable accuracy
THRESHOLD = 0.70

if accuracy >= THRESHOLD:
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")
    raise SystemExit(1)
