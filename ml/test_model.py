'''
Unit test for function in model.py
'''
import numpy as np
import pandas as pd
from ml.model import train_model, inference, compute_model_metrics
import pytest
from pytest_mock import mocker


@pytest.fixture()
def X_train():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture()
def X_test():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture()
def y_train():
    return pd.Series([0, 1, 1])


@pytest.fixture()
def y():
    return y_train


@pytest.fixture()
def preds():
    return y_train


@pytest.fixture()
def test_model(preds):
    class MockModel:
        def __init__(self, preds=preds):
            self.preds = preds

        def predict(self, X):
            return self.preds

    return MockModel()


def test_train_model(mocker, X_train, y_train):
    mock_model = mocker.patch("ml.model.LogisticRegression")

    _ = train_model(X_train, y_train)

    mock_model.assert_called()
    mock_model().fit.assert_called_once_with(X_train, y_train)


def test_compute_model_metrics(mocker, y, preds):
    mock_fbeta_score = mocker.patch("ml.model.fbeta_score")
    mock_precision_score = mocker.patch("ml.model.precision_score")
    mock_recall_score = mocker.patch("ml.model.recall_score")

    _ = compute_model_metrics(y, preds)

    mock_fbeta_score.assert_called_once_with(y, preds, beta=1, zero_division=1)
    mock_precision_score.assert_called_once_with(y, preds, zero_division=1)
    mock_recall_score.assert_called_once_with(y, preds, zero_division=1)


def test_inference(test_model, X_test, preds):

    result = inference(test_model, X_test)
    assert result == preds
