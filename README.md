
Housing Price Prediction Using Linear Regression

This project uses Linear Regression to predict the housing prices in California based on various features such as median income, house age, and average rooms. The model is trained on the California Housing Dataset and evaluates the prediction performance using the Mean Squared Error (MSE). Additionally, the model is saved for future use and can be used to make predictions for new data.

Prerequisites
You will need to install the following Python libraries:

numpy
pandas
scikit-learn
joblib

Dataset
The dataset used in this project is the California Housing Dataset, which contains data about housing prices and several related features. These features include:

MedInc: Median income
HouseAge: Age of the house
AveRooms: Average number of rooms
AveBedrms: Average number of bedrooms
Population: Population of the area
AveOccup: Average occupancy
Latitude: Latitude of the house
Longitude: Longitude of the house
MEDV: Median house value (target variable)
The dataset is fetched from scikit-learn's fetch_california_housing() function.

Steps
1. Data Loading:
The California housing data is loaded and converted into a pandas DataFrame. The target variable MEDV (Median House Value) is added as a column to the dataset.

2. Data Preprocessing:
The data is split into training and testing sets using train_test_split(). The features are scaled using StandardScaler() for better performance in the regression model.

3. Model Pipeline:
A preprocessing pipeline is created to standardize the numerical features, and then a Linear Regression model is applied on the preprocessed data using make_pipeline().

4. Model Training:
The model is trained on the training set (x_train, y_train).

5. Model Evaluation:
The model's performance is evaluated on the test set (x_test, y_test) using Mean Squared Error (MSE). The MSE gives a measure of how well the model predicts the housing prices.

6. Model Saving:
After training, the model is saved to a file using joblib for later use.

7. Prediction for New Data:
An example prediction is made for a new house with given features. The saved model is loaded and used to predict the house price based on the features of the new house.


Example Usage
After training the model, you can use it to predict housing prices for new data. For example, to predict the price of a new house, you can define the features for the house and use the saved model to make predictions.

python
Copy code
new_house = pd.DataFrame({
    'MedInc': [3.0],
    'HouseAge': [20.0],
    'AveRooms': [5.0],
    'AveBedrms': [2.0],
    'Population': [1000.0],
    'AveOccup': [3.0],
    'Latitude': [37.5],
    'Longitude': [-122.5],
})

loaded_model = joblib.load('housing_prices_model.joblib')
predicted_price = loaded_model.predict(new_house) * 100000  # Scale the price
print(f'Predicted Price for the New House: ${predicted_price[0]:.2f}')
File Structure
housing_prices_model.joblib: The saved model.
housing_price_prediction.py: The script used for training, evaluation, and prediction.

Notes
The model expects only numeric features in the input. For more complex use cases with mixed data types (e.g., text), you can extend the pipeline to include transformers for text data (e.g., using CountVectorizer).
The target variable (MEDV) is scaled by 100,000 during prediction to match the typical housing price scale.
