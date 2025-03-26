import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Explicitly define Mean Squared Error (mse) when loading the model
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
model = tf.keras.models.load_model("tf_bridge_model.h5", custom_objects=custom_objects)

# Load preprocessing pipeline
with open("preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Define input features
feature_columns = ["Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes", "Material", "Condition_Rating"]

# Predefined material categories (as assumed or from your dataset)
material_categories = ["Steel", "Concrete", "Composite"]

# Streamlit UI
st.title("Bridge Load Capacity Predictor")

# User inputs
span_ft = st.number_input("Bridge Span (ft)", min_value=1, value=250)
deck_width_ft = st.number_input("Deck Width (ft)", min_value=1, value=40)
age_years = st.number_input("Bridge Age (years)", min_value=0, value=20)
num_lanes = st.number_input("Number of Lanes", min_value=1, value=2)
material = st.selectbox("Bridge Material", material_categories)
condition_rating = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, value=4)

# Create DataFrame with correct feature order
input_data = pd.DataFrame([[span_ft, deck_width_ft, age_years, num_lanes, material, condition_rating]],
                          columns=feature_columns)

# Apply preprocessing
try:
    input_data_transformed = preprocessor.transform(input_data)
except ValueError as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

# Predict maximum load capacity
if st.button("Predict Load Capacity"):
    prediction = model.predict(input_data_transformed)[0][0]
    st.success(f"Estimated Maximum Load Capacity: **{prediction:.2f} tons**")

