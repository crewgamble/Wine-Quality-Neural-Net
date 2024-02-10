# Wine Quality Prediction

## Project Overview
This project uses a neural network to predict the quality of wine based on various physicochemical features. The model is built using PyTorch and trained on the "Wine Quality" dataset available from the UCI Machine Learning Repository. The aim is to demonstrate how to preprocess data, implement a regression model in PyTorch, and evaluate its performance.

## Features
- Data normalization and preprocessing
- Implementation of a regression model using PyTorch's neural network module
- Model evaluation using Root Mean Squared Error (RMSE)
- Visualization of actual vs predicted wine quality

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6+
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib

You can install the required libraries using the following command:
```
pip install torch pandas scikit-learn matplotlib
```

## Dataset
The dataset used is the "Wine Quality" dataset, focusing on red variants of the Portuguese "Vinho Verde" wine. It can be automatically downloaded and loaded into the program using the provided URL in the code.

## Installation
To run this project, clone the repo to your local machine using:
```
git clone https://github.com/yourusername/wine-quality-prediction.git
```

## Usage
Navigate to the project directory and run the script using Python:
```
cd wine-quality-prediction
python wine_quality_prediction.py
```
