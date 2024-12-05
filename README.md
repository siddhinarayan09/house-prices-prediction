# Housing Prices Prediction Project

This project predicts housing prices based on various features such as median income, house age, and average number of rooms. It leverages machine learning techniques with a Linear Regression model and provides insights through evaluation metrics.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Output](#output)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## Overview

The **Housing Prices Prediction Project** applies machine learning to analyze the **California Housing Dataset** and predict housing prices. It preprocesses data, trains a model, evaluates performance, and saves the model for future predictions.

---

## Features

- **Data Preprocessing**: 
  - Standardizes numerical features using `StandardScaler` for consistent model training.
- **Model Training**: 
  - A pipeline integrates preprocessing with a Linear Regression model.
- **Model Evaluation**: 
  - Calculates Mean Squared Error (MSE) to assess model accuracy.
- **Model Persistence**: 
  - Saves the trained model with `joblib` for reuse.
- **Prediction**: 
  - Accepts new input data and predicts housing prices.

---

## Technologies Used

- **Python** (Core Language)
- **NumPy** and **pandas** (Data Handling)
- **scikit-learn** (Modeling, Preprocessing, Evaluation)
- **joblib** (Model Serialization)

---

## How It Works

1. **Dataset**:
   - The **California Housing Dataset** is loaded using `fetch_california_housing`.
   - Features and target values (median house prices) are extracted.
   
2. **Data Splitting**:
   - The dataset is split into training (80%) and testing (20%) subsets.

3. **Preprocessing**:
   - Numerical features are standardized using `StandardScaler` within a `ColumnTransformer`.

4. **Model Training**:
   - A Linear Regression model is trained using the preprocessed training data.

5. **Evaluation**:
   - The model predicts prices for the test set, and the Mean Squared Error (MSE) is computed.

6. **Saving and Loading the Model**:
   - The trained model is saved as `housing_prices_model.joblib` for future predictions.
   - The saved model is reloaded for predicting prices for new data.

---

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/housing-prices-prediction.git
   cd housing-prices-prediction
2. **Install Dependencies**:
   Ensure Python 3.6+ is installed, then install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn joblib
3. **Run the Script**:
   Execute the Python script:
   ```bash
   python main.py

## Usage
**Run the Script**:

Train the model, evaluate its performance, and save it for reuse.

**Predict New Prices**:

Modify the new_house DataFrame in the script with the desired input features.
Load the saved model and make predictions for the new house.

## Output
**Mean Squared Error**: Evaluates model accuracy on test data.
```bash
Mean Squared Error on Test Data: 0.47
Predicted Price for the New House: $237500.00



