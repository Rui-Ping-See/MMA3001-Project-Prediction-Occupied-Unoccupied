"""
MMA3001 Occupancy Prediction Project.
This module contains functions used to prepare data, train machine-learning models, and evaluate occupancy predictions.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# 1. calculate dectection rate
def calculate_detection_rates(y_true, y_pred):
    """
    Calculate occupied and unoccupied detection rates.
    Parameters:
    y_true: array
        Actual occupancy values, where 1 = occupied and 0 = unoccupied.
    y_pred: array
        Predicted occupancy values, where 1 = occupied and 0 = unoccupied.

    Returns:
    occupied_rate: float
        Percentage of occupied samples correctly detected.
    unoccupied_rate: float
        Percentage of unoccupied samples correctly detected.
    """

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    occupied_rate = (TP / (TP + FN)) * 100
    unoccupied_rate = (TN / (TN + FP)) * 100

    return occupied_rate, unoccupied_rate

# 2. calculate balanced accuracy
def calculate_balanced_accuracy(occupied_rate, unoccupied_rate):
    """
    Calculate balanced accuracy from the occupied and unoccupied detection rates.
    Parameters:
    occupied_rate: float
        Percentage of occupied samples correctly detected.
    unoccupied_rate: float
        Percentage of unoccupied samples correctly detected.

    Returns:
    balanced_accuracy: float
        Average of the occupied and unoccupied detection rates.
    """

    balanced_accuracy = (occupied_rate + unoccupied_rate) / 2

    return balanced_accuracy

# 3. run Linear Regression
def run_linear_regression(x_train, y_train, x_test):
    """
    Train a Linear Regression model and predict occupancy values.

    Parameters:
    x_train: array
        Training input data containing CO2, temperature and humidity.
    y_train: array
        Training occupancy values.
    x_test: array
        Testing input data used to make predictions.

    Returns:
    model: LinearRegression
        Trained Linear Regression model.
    y_pred: array
        Predicted occupancy values for the test data.
    """

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return model, y_pred

# 4. run KNN
def run_knn(x_train, y_train, x_test, k=1):
    """
    Train a KNN model and predict occupancy values.

    Parameters:
    x_train: array
        Training input data containing CO2, temperature and humidity.
    y_train: array
        Training occupancy values.
    x_test: array
        Testing input data used to make predictions.
    k : int, optional
        Number of nearest neighbours used by KNN. Default is 1.

    Returns:
    model: Pipeline
        Trained KNN pipeline including feature scaling.
    y_pred: array
        Predicted occupancy values for the test data.
    """

    model = Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=k))])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return model, y_pred
