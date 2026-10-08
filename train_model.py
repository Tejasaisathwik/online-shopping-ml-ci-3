import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Load dataset
data = pd.read_csv("data/online_shopping_purchase_prediction_raw.csv")

print("Dataset loaded successfully")
print("Rows:", len(data))
print("Columns:", len(data.columns))


# Target column
target = "Purchased"

X = data.drop(columns=[target])
y = data[target]


# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()


# Preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)


# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train
pipeline.fit(X_train, y_train)


# Prediction
y_pred = pipeline.predict(X_test)


# Metrics
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
