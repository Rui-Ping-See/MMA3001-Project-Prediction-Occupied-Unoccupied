import numpy as np

from src.occupancy_prediction import (calculate_detection_rates, calculate_balanced_accuracy, run_linear_regression, run_knn)


def test_detection_rates():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])

    occupied_rate, unoccupied_rate = calculate_detection_rates(y_true, y_pred)

    assert occupied_rate == 50
    assert unoccupied_rate == 50


def test_balanced_accuracy():
    occupied_rate = 80
    unoccupied_rate = 60

    result = calculate_balanced_accuracy(occupied_rate, unoccupied_rate)

    assert result == 70


def test_linear_regression():
    x_train = np.array([[400, 22, 40], [500, 23, 45], [600, 24, 50], [700, 25, 55]])

    y_train = np.array([0, 0, 1, 1])

    x_test = np.array([[450, 22.5, 42], [650, 24.5, 52]])

    model, y_pred = run_linear_regression(x_train, y_train, x_test)

    assert len(y_pred) == 2


def test_knn():
    x_train = np.array([[400, 22, 40], [500, 23, 45], [600, 24, 50], [700, 25, 55]])

    y_train = np.array([0, 0, 1, 1])

    x_test = np.array([[410, 22, 40], [690, 25, 55]])

    model, y_pred = run_knn(x_train, y_train, x_test, k=1)

    assert len(y_pred) == 2
    assert y_pred[0] == 0
    assert y_pred[1] == 1
