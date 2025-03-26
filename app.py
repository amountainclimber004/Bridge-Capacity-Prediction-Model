import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

# --- Load Saved Artifacts ---

# Load preprocessing pipeline
with open('preprocessing/preprocessing_pipeline.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load trained TensorFlow model
model = tf.keras.models.load_model('models/tf_bridge_model.h5')

# Optionally, load the CSV to show a sample of the data (for reference)
data_path = 'data/lab_11_bridge_data.csv'
df = pd.read_csv(data_path)

# --- Streamlit App ---

st.title("Bridge Maximum Load Prediction")

st.markdown("""
This web app predicts the maximum load capacity (in tons) of a bridge based on its characteristics.
Please enter the bridge details below:
""")

# --- User Inputs ---

col1, col2 = st.columns(2)
with col1:
    span_ft = st.number_input("Span (ft)", min_value=0.0, value=250.0)
    deck_width_ft = st.number_input("Deck Width (ft)", min_value=0.0, value=40.0)
    age_years = st.number_input("Age (Years)", min_value=0, value=20)
with col2:
    num_lanes = st.number_input("Number of Lanes", min_value=1, value=4)
    condition_rating = st.number_input("Condition Rating (1-5)", min_value=1, max_value=5, value=4)
    material = st.selectbox("Material", options=["Steel", "Concrete", "Composite"])

# Prepare a DataFrame for the user input
input_data = pd.DataFrame({
    'Span_ft': [span_ft],
    'Deck_Width_ft': [deck_width_ft],
    'Age_Years': [age_years],
    'Num_Lanes': [num_lanes],
    'Condition_Rating': [condition_rating],
    'Material': [material]
})

# --- Display Sample Data ---
st.markdown("### Sample Data")
if st.checkbox("Show sample data from the dataset"):
    st.dataframe(df.head(10))

# --- Prediction ---
if st.button("Predict Maximum Load"):
    # Preprocess the user input using the saved pipeline
    input_processed = preprocessor.transform(input_data)
    
    # Make prediction with the loaded model
    prediction = model.predict(input_processed)
    predicted_load = prediction[0][0]
    
    st.success(f"Predicted Maximum Load: {predicted_load:.2f} Tons")

st.markdown("""
---
**Note:** This model and dataset are created for educational purposes only.
""")
