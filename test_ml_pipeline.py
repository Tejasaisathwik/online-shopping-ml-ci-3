import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
df = pd.read_csv("online_shopping_purchase_prediction_raw.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# Target column
target = "Purchased"

X = df.drop(columns=[target])
y = df[target]


# Categorical columns
categorical_columns = [
    "Gender",
    "DiscountUsed",
    "DeviceType",
    "Membership"
]


# Numerical columns
numerical_columns = [
    "Age",
    "MonthlyIncome",
    "WebsiteVisits",
    "PagesViewed",
    "TimeSpentMinutes",
    "PreviousPurchases",
    "CartItems",
    "CustomerRating",
    "PurchaseAmount"
]


# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training model...")


# Train model
pipeline.fit(X_train, y_train)


# Predictions
y_pred = pipeline.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Save trained model
MODEL_FILE = "online_shopping_purchase_model.pkl"

joblib.dump(pipeline, MODEL_FILE)

print("Model saved successfully:")
print(MODEL_FILE)


# Save metrics
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully.")


# Verify model file
import os

if os.path.exists(MODEL_FILE):
    print("SUCCESS: Model file exists.")
    print("Model file size:", os.path.getsize(MODEL_FILE), "bytes")
else:
    print("ERROR: Model file was not created.")
    raise SystemExit(1)
