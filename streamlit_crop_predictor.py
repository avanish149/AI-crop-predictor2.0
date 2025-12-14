import random
import os
import pandas as pd
import streamlit as st
import pickle

# App title and instructions
st.title("AI Crop Predictor Dashboard")
st.write("Upload your dataset, view the data, and try out crop predictions interactively!")

# ---------------------------
# 1) Load data
# ---------------------------
DATAFILE = "crop_recommendation.csv"
if not os.path.isfile(DATAFILE):
    st.error(f"Dataset file '{DATAFILE}' not found in the current directory.")
    st.stop()

data = pd.read_csv(DATAFILE)

# 2) Rename columns if needed
column_map = {
    "Nitrogen": "N",
    "Phosphorus": "P",
    "Potassium": "K",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "pH_Value": "ph",
    "Rainfall": "rainfall",
    "Crop": "label"
}
data = data.rename(columns=column_map)

# NOW also require rates and yield (added to CSV earlier)
required_columns = {
    "N", "P", "K", "temperature", "humidity",
    "ph", "rainfall", "label", "rates", "yield"
}
missing = required_columns - set(data.columns)
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

# ---------------------------
# 3) Display DataFrame
# ---------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(100))  # Show first 100 for quick view

# ---------------------------
# 4) User Input Section
# ---------------------------
st.subheader("Enter Values to Predict Crop")

# State to store randomized values
if "rand_values" not in st.session_state:
    st.session_state["rand_values"] = {
        "N": 50,
        "P": 50,
        "K": 50,
        "temperature": 25.0,
        "humidity": 50.0,
        "ph": 7.0,
        "rainfall": 100.0,
    }

def randomize_inputs():
    st.session_state["rand_values"] = {
        "N": random.randint(0, 200),
        "P": random.randint(0, 200),
        "K": random.randint(0, 200),
        "temperature": round(random.uniform(0, 60), 2),
        "humidity": round(random.uniform(0, 100), 2),
        "ph": round(random.uniform(0, 14), 2),
        "rainfall": round(random.uniform(0, 400), 2),
    }

# Button to trigger randomization
if st.button("Randomize Inputs"):
    randomize_inputs()

with st.form("prediction_form"):
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200,
                        value=st.session_state["rand_values"]["N"])
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200,
                        value=st.session_state["rand_values"]["P"])
    K = st.number_input("Potassium (K)", min_value=0, max_value=200,
                        value=st.session_state["rand_values"]["K"])
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0,
                                  value=st.session_state["rand_values"]["temperature"])
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0,
                               value=st.session_state["rand_values"]["humidity"])
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0,
                         value=st.session_state["rand_values"]["ph"])
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0,
                               value=st.session_state["rand_values"]["rainfall"])
    submit = st.form_submit_button("Predict Crop")

# Fixed defaults for economic features (same for all predictions for now)
default_rates = 25.5
default_yield = 3850

# ---------------------------
# 5) Load trained model and predict
# ---------------------------
MODELFILE = "crop_model.pkl"
if not os.path.isfile(MODELFILE):
    st.warning(f"No trained model found (expected {MODELFILE}). Please train your model and place the file here.")
else:
    with open(MODELFILE, "rb") as f:
        model = pickle.load(f)

    if submit:
        # Order must match training: N,P,K,temperature,humidity,ph,rainfall,rates,yield
        input_df = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall, default_rates, default_yield]],
            columns=["N", "P", "K", "temperature", "humidity",
                     "ph", "rainfall", "rates", "yield"]
        )
        pred = model.predict(input_df)[0]
        st.success(f"Recommended Crop: {pred}")

# ---------------------------
# 6) Basic statistics
# ---------------------------
st.subheader("Basic Dataset Statistics")
st.write(data.describe())


