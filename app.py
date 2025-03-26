import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load preprocessing pipeline
with open("preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model("tf_bridge_model.h5")

# Define material categories (ensure this matches training order)
MATERIAL_CATEGORIES = ['Steel', 'Concrete', 'Composite']

# Streamlit app title
st.title("Bridge Load Capacity Predictor")

# User input fields
span_ft = st.number_input("Bridge Span (ft)", min_value=1, value=250)
deck_width_ft = st.number_input("Deck Width (ft)", min_value=1, value=40)
age_years = st.number_input("Bridge Age (years)", min_value=0, value=20)
num_lanes = st.number_input("Number of Lanes", min_value=1, value=2)
material = st.selectbox("Bridge Material", MATERIAL_CATEGORIES)
condition_rating = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, value=4)

# Convert categorical material input to match preprocessing pipeline
material_encoded = [1 if material == m else 0 for m in MATERIAL_CATEGORIES]  # One-hot encoding

# Prepare input data
input_data = np.array([[span_ft, deck_width_ft, age_years, num_lanes, *material_encoded, condition_rating]])

# Apply preprocessing
input_data_transformed = preprocessor.transform(input_data)

# Predict max load capacity
if st.button("Predict Load Capacity"):
    prediction = model.predict(input_data_transformed)[0][0]
    st.success(f"Estimated Maximum Load Capacity: **{prediction:.2f} tons**")



