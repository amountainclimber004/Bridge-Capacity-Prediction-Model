import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import os

# Ensure proper working directory
st.write("Current working directory:", os.getcwd())

# Define custom loss function dictionary for loading the model
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load Preprocessing Pipeline
preprocessing_path = "preprocessing_pipeline.pkl"
if os.path.exists(preprocessing_path):
    with open(preprocessing_path, 'rb') as f:
        preprocessor = pickle.load(f)
else:
    st.error(f"Error: Preprocessing file not found at {preprocessing_path}")

# Load Model
model_path = "tf_bridge_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
else:
    st.error(f"Error: Model file not found at {model_path}")

# Load Sample Data (optional, for reference)
data_path = "lab_11_bridge_data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.warning(f"Warning: Dataset file not found at {data_path}")

# Streamlit App UI
st.title("Bridge Load Capacity Prediction")
st.write("Enter bridge parameters below to predict its maximum load capacity.")

# User Inputs
span_ft = st.number_input("Bridge Span (ft)", min_value=50, max_value=1000, value=250)
deck_width_ft = st.number_input("Deck Width (ft)", min_value=10, max_value=100, value=40)
age_years = st.number_input("Age of Bridge (years)", min_value=0, max_value=150, value=20)
num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, value=2)

# Material Selection
materials = ['Steel', 'Concrete', 'Composite']
material = st.selectbox("Bridge Material", materials)
material_encoded = [1 if material == m else 0 for m in materials]  # One-hot encoding

condition_rating = st.slider("Condition Rating (1 - Poor to 5 - Excellent)", min_value=1, max_value=5, value=3)

# Prepare Input Data for Prediction
input_data = np.array([[span_ft, deck_width_ft, age_years, num_lanes, *material_encoded, condition_rating]])
input_data_transformed = preprocessor.transform(input_data)  # Apply the same preprocessing as during training

# Prediction Button
if st.button("Predict Load Capacity"):
    prediction = model.predict(input_data_transformed)
    st.success(f"Estimated Maximum Load Capacity: {prediction[0][0]:.2f} tons")

