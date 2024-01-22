import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Get the current working directory
project_directory = os.getcwd()

# Define the paths to the dataset and the model
dataset_path = os.path.join(project_directory, 'dataset', 'repair_cost_dataset.csv')
model_path = os.path.join(project_directory, 'models', 'linear_regression_model.pkl')

# Load the dataset
df = pd.read_csv(dataset_path)

# One-hot encode the 'Damaged_Parts' and 'Severity' columns
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_parts_severity = encoder.fit_transform(df[['Damaged_Parts', 'Severity']])

# Get the feature names after one-hot encoding
feature_names = encoder.get_feature_names_out(['Damaged_Parts', 'Severity'])

# Create a DataFrame with one-hot encoded features
df_encoded = pd.concat([df, pd.DataFrame(encoded_parts_severity, columns=feature_names)], axis=1)

# Select features and target variable
X = df_encoded.drop(['Damaged_Parts', 'Severity', 'Cost_Repair_Euros'], axis=1)
y = df_encoded['Cost_Repair_Euros']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(model, model_path)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example: Make a prediction for a new data point
new_damaged_parts = 'Bumper, Fender, Door'
new_severity = 'severe'
new_data = pd.DataFrame([[new_damaged_parts, new_severity]], columns=['Damaged_Parts', 'Severity'])

# One-hot encode the new data
new_encoded = encoder.transform(new_data)

# Predict the repair cost
predicted_cost = model.predict(new_encoded)
print(f'Predicted Cost Repair (Euros): {predicted_cost[0]}')
