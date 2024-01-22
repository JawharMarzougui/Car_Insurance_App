import streamlit as st
import os
import numpy as np
import base64
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import OneHotEncoder
from roboflow import Roboflow
import pandas as pd
import joblib

# Get the current working directory
project_directory = os.getcwd()

# Define the paths to the dataset and the model
dataset_path = "C:/Users/jawha/OneDrive/Desktop/Car_Insurrance_Project/dataset/repair_cost_dataset.csv"

# Get the path to the 'models' folder
models_folder = os.path.join(os.path.dirname(__file__), 'models')

# Load the dataset
df = pd.read_csv(dataset_path)

# Load the linear regression model
linear_regression_model_path = os.path.join(models_folder, 'linear_regression_model.pkl')
linear_regression_model = joblib.load(linear_regression_model_path)

# Function to get one-hot encoded features for linear regression
def get_one_hot_encoded_features(damaged_parts, severity):
    # Create a DataFrame with a single row for the input data
    input_data = pd.DataFrame([[damaged_parts, severity]], columns=['Damaged_Parts', 'Severity'])

    # One-hot encode the input data
    encoded_features = encoder.transform(input_data[['Damaged_Parts', 'Severity']])

    return encoded_features

# Get the path to the 'weights' folder
weights_folder = os.path.join(os.path.dirname(__file__), 'weights')

# Load the real or fake classifier model
real_or_fake_model_path = os.path.join(weights_folder, 'image_classifier_model.h5')
real_or_fake_model = load_model(real_or_fake_model_path)

# Load the car damage classifier model
car_damage_model_path = os.path.join(weights_folder, 'fine_tuned_car_damage_classifier.h5')
car_damage_model = load_model(car_damage_model_path)

# Load the damaged parts detection model from Roboflow
rf = Roboflow(api_key="Tc2h5YtqzXnEyY01651e")
project = rf.workspace().project("car-damage-detection-t0g92")
model = project.version(3).model

# Define the prediction function for real or fake classifier
def predict_real_or_fake(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = real_or_fake_model.predict(img_array)
    return prediction[0][0]

# Define the prediction function for car damage classifier
def predict_car_damage(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    prediction = car_damage_model.predict(img_array)
    return prediction[0]

# Streamlit app
def main():
    st.title("Car Insurance App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.read())

        # Make predictions
        real_or_fake_prediction = predict_real_or_fake(temp_path)
        car_damage_prediction = predict_car_damage(temp_path)

        # Interpret the predictions
        real_or_fake_result = "Real Image" if real_or_fake_prediction > 0.5 else "AI Generated Image"
        car_damage_classes = ['minor', 'moderate', 'severe']
        car_damage_result = car_damage_classes[np.argmax(car_damage_prediction)]

        # Make prediction with Roboflow model
        prediction_output = model.predict(temp_path, confidence=40, overlap=30)
        prediction_image_path = "prediction.jpg"
        prediction_output.save(prediction_image_path)

        # Display the prediction image
        st.image(prediction_image_path, width=600)

        # Display the results
        st.write("   ")
        st.write("   ")
        st.markdown(f"**Real or Fake:** {real_or_fake_result} (Confidence: {real_or_fake_prediction:.2f})")
        st.markdown(f"**Car Damage Severity:** {car_damage_result} (Confidence: {max(car_damage_prediction):.2f})")

        # Get damaged parts and severity
        prediction_output = model.predict(temp_path, confidence=40, overlap=30).json()
        severity = car_damage_result

        L = []
        for prediction in prediction_output['predictions']:
            label = prediction.get("class", "")
            L.append(label)

        damaged_parts = ', '.join(map(str, L))

        # One-hot encode the 'Damaged_Parts' and 'Severity' columns
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_parts_severity = encoder.fit_transform(df[['Damaged_Parts', 'Severity']])

        # Get the feature names after one-hot encoding
        feature_names = encoder.get_feature_names_out(['Damaged_Parts', 'Severity'])

        # Create a DataFrame with one-hot encoded features
        df_encoded = pd.concat([df, pd.DataFrame(encoded_parts_severity, columns=feature_names)], axis=1)

        new_data = pd.DataFrame([[damaged_parts, severity]], columns=['Damaged_Parts', 'Severity'])
        # One-hot encode the new data
        new_encoded = encoder.transform(new_data)

        # Predict the repair cost
        predicted_cost = linear_regression_model.predict(new_encoded)
        formatted_result = "{:.2f}".format(abs(predicted_cost[0]))


        # Display the results with bold formatting
        st.markdown(f"**Damaged Parts:** {damaged_parts}")
        st.markdown(f"**Predicted Cost Repair:** {formatted_result} euros")

        
        # Remove the temporary file
        os.remove(temp_path)

if __name__ == "__main__":
    main()
