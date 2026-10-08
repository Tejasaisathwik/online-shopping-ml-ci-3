import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Check current folder
print("Current folder:", os.getcwd())
print("Files in project:", os.listdir("."))

# Check data folder
print("Files in project:", os.listdir("."))

data = pd.read_csv("online_shopping_purchase_prediction_raw.csv")

print("Dataset loaded successfully")
print("Rows:", len(data))
print("Columns:", len(data.columns))

# Target column
target = "Purchased"

X = data.drop(columns=[target])
y = data[target]

# Identify columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)


# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training model...")

# Train
pipeline.fit(X_train, y_train)


print("Model training completed")
joblib.dump(pipeline, "online_shopping_model.pkl")

print("Model saved as online_shopping_model.pkl")


# Prediction
y_pred = pipeline.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))


# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}


with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


print("metrics.json created successfully")
