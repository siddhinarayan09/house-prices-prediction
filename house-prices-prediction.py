import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler 
import joblib
from sklearn.datasets import fetch_california_housing

#downloading the dataset
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns = california_housing.feature_names)

data['MEDV'] = california_housing.target

#data.head(10)

#splittimg the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data.drop('MEDV', axis=1), data['MEDV'], test_size = 0.2, random_state = 42 )

#creating the preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x_train.columns),
        #('text', CountVectorizer(), 'Description')
    ]
)

#combining the pipeline with the model
model = make_pipeline(preprocessor, LinearRegression())

#trsining the model
model.fit(x_train, y_train)

#evaluating the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

#save the model for future use
joblib.dump(model, 'housing_prices_model.joblib')

# Example: Predict the price for a new house
new_house = pd.DataFrame({
    'MedInc': [3.0],  # Example numerical features, use appropriate values from the dataset
    'HouseAge': [20.0],
    'AveRooms': [5.0],
    'AveBedrms': [2.0],
    'Population': [1000.0],
    'AveOccup': [3.0],
    'Latitude': [37.5],
    'Longitude': [-122.5],
    #'Description': ["Charming cottage with a garden"]  # Example text feature
})

# Load the pre-trained model
loaded_model = joblib.load('housing_prices_model.joblib')

# Make predictions for the new house
predicted_price = loaded_model.predict(new_house) * 100000
print(f'Predicted Price for the New House: ${predicted_price[0]:.2f}')