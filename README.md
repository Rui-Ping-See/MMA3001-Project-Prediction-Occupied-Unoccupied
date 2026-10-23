# MMA3001 Project - Room Occupancy Prediction 
This project investigates whether environmental sensor measurements can be used to predict whether a room is occupied or unoccupied.

## Engineering Problem:
Can environmental sensor measurements be used to predict whether a room is occupied or unoccupied?

## Inputs:
The environmental measurements used are:
1. CO2 concentration 
2. Temperature
3. Humidity

## Output:
The output is room occupancy status:
1 = Occupied
0 = Unoccupied

## Methods:
Two machine-learning methods were used:
1. Linear Regression
2. K-Nearest Neighbours Regression (KNN) 
The methods were compared using overall accuracy, occupied detection rate, unoccupied detection rate, balanced accuracy and computational runtime.

## Repository Structure:
1. Project_3001.ipynb - main Google Colab notebook containing the full complete analysis
2. src/occupancy_prediction.py - reusable Python functions used for model training and evaluation
3. tests/test_occupancy_prediction.py - pytest test functions
4. tests/test_report.txt - pytest test results
5. requirements.txt - Python libraries required to run the project
6. LICENSE: MIT License

## Required Python Libraries:
The Python libraries required to run the project are listed in requirements.txt.
Install them using: pip install -r requirements.txt

## Testing:
The source-code functions are tested using pytest.
Run the tests using: PYTHONPATH=. pytest
All four tests passed successfully.

## Main Notebook:
The complete data preparation, model development, validation, sensitivity analysis, model comparison and computational analysis are provided in Project_3001.ipynb.
