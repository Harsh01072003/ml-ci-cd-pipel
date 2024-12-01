import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# model/train.py


# Load dataset
data = pd.read_csv("D:/ml-ci-cd-pipe/data/Boston.csv")

# Split dataset
X = data.drop('MEDV', axis=1)
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/house_price_model.pkl')

# Create the 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the model
joblib.dump(model, 'model/house_price_model.pkl')

print("Model saved successfully!")