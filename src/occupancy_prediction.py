"""
MMA3001 Occupancy Prediction Project.

This module contains functions used to prepare data, train machine-learning models, and evaluate occupancy predictions.
"""

import numpy as np

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

def calculate_balanced_accuracy(occupied_rate, unoccupied_rate):
    """
    Calculate balanced accuracy from the occupied and unoccupied detection rates.
    Parameters:
    occupied_rate : float
        Percentage of occupied samples correctly detected.
    unoccupied_rate : float
        Percentage of unoccupied samples correctly detected.

    Returns:
    balanced_accuracy : float
        Average of the occupied and unoccupied detection rates.
    """

    balanced_accuracy = (occupied_rate + unoccupied_rate) / 2

    return balanced_accuracy
