# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
from joblib import dump, load

data = pd.read_csv('../data/census.csv')

# Optional enhancement, use K-fold cross
# validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
dump(model, "../model/model.joblib")
dump(encoder, "../model/encoder.joblib")
dump(lb, "../model/lb.joblib")

saved_model = load("../model/model.joblib")


y_pred_test = inference(saved_model, X_test)
print(compute_model_metrics(y_test, y_pred_test))
