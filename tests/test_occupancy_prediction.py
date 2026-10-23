import numpy as np

from src.occupancy_prediction import (calculate_detection_rates, calculate_balanced_accuracy)


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
