import random
import os
import pandas as pd
import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------
# 1) Page setup
# -------------------------------------------------
st.title("AI Crop Predictor Dashboard")
st.write(
"This dashboard uses a machine‑learning model trained on soil nutrients, "
"weather conditions, and historical crop records to recommend the most "
"suitable crop for your field. Enter values for nitrogen, phosphorus, "
"potassium, temperature, humidity, pH, and rainfall to see the suggested "
"crop along with an estimated market rate and expected yield."
)


DATAFILE = "crop_recommendation.csv"
MODELFILE = "crop_model.pkl"

# -------------------------------------------------
# 2) Load dataset
# -------------------------------------------------
if not os.path.isfile(DATAFILE):
st.error(f"Dataset file '{DATAFILE}' not found in the current directory.")
st.stop()

data = pd.read_csv(DATAFILE)

# Standardize column names
column_map = {
"Nitrogen": "N",
"Phosphorus": "P",
"Potassium": "K",
"Temperature": "temperature",
"Humidity": "humidity",
"pH_Value": "ph",
"Rainfall": "rainfall",
"Crop": "label",
}
data = data.rename(columns=column_map)

required_columns = {
"N", "P", "K", "temperature", "humidity",
"ph", "rainfall", "label", "rates", "yield",
}
missing = required_columns - set(data.columns)
if missing:
st.error(f"Dataset is missing columns: {missing}")
st.stop()

# -------------------------------------------------
# 3) Load pre-trained model (no training in app)
# -------------------------------------------------
if not os.path.isfile(MODELFILE):
st.error(
f"Model file '{MODELFILE}' not found. "
f"Run your training script (crop_predictor_safe.py) first."
)
st.stop()

@st.cache_resource
def load_model(path: str):
with open(path, "rb") as f:
return pickle.load(f)

model = load_model(MODELFILE)

st.subheader("Model status")
# Set this to the accuracy you observed when running crop_predictor_safe.py
st.write("RandomForest loaded from file.")
st.write("Offline test accuracy: **99.32%**")

# -------------------------------------------------
# 4) Dataset preview
# -------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(100))

# -------------------------------------------------
# 5) User input section
# -------------------------------------------------
st.subheader("Enter Values to Predict Crop")

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

if st.button("Randomize Inputs"):
randomize_inputs()

with st.form("prediction_form"):
N = st.number_input("Nitrogen (N)", 0, 200, st.session_state["rand_values"]["N"])
P = st.number_input("Phosphorus (P)", 0, 200, st.session_state["rand_values"]["P"])
K = st.number_input("Potassium (K)", 0, 200, st.session_state["rand_values"]["K"])
temperature = st.number_input(
"Temperature (°C)", 0.0, 60.0, st.session_state["rand_values"]["temperature"]
)
humidity = st.number_input(
"Humidity (%)", 0.0, 100.0, st.session_state["rand_values"]["humidity"]
)
ph = st.number_input(
"pH Value", 0.0, 14.0, st.session_state["rand_values"]["ph"]
)
rainfall = st.number_input(
"Rainfall (mm)", 0.0, 400.0, st.session_state["rand_values"]["rainfall"]
)
submit = st.form_submit_button("Predict Crop")

# Fixed defaults for economic features used at display time
default_rates = 25.5
default_yield = 3850

# Mapping crop -> (rate, yield)
crop_data = {
"rice": (25.5, 3850),
"maize": (18.2, 4200),
"chickpea": (65.0, 850),
"kidneybeans": (45.0, 2800),
"pigeonpeas": (70.0, 720),
"mothbeans": (55.0, 450),
"mungbean": (60.0, 500),
"blackgram": (58.0, 480),
"lentil": (62.0, 950),
"pomegranate": (80.0, 22000),
"banana": (35.0, 35000),
"mango": (45.0, 8500),
"grapes": (120.0, 22000),
"watermelon": (12.0, 25000),
"muskmelon": (15.0, 28000),
"apple": (150.0, 20000),
"orange": (40.0, 15000),
"papaya": (25.0, 35000),
"coconut": (30.0, 14000),
"cotton": (120.0, 800),
"jute": (35.0, 2500),
"coffee": (200.0, 1200),
}

# -------------------------------------------------
# 6) Prediction using loaded model
# -------------------------------------------------
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
# Use same feature order as during training
default_rates = 25.5
default_yield = 3850.0
cols = ["N", "P", "K", "temperature", "humidity", "ph",
"rainfall", "rates", "yield"]
arr = [[N, P, K, temperature, humidity, ph,
rainfall, default_rates, default_yield]]
df = pd.DataFrame(arr, columns=cols)
pred = model.predict(df)[0]
return pred


if submit:
pred = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
crop_key = str(pred).strip().lower()
rate, yld = crop_data.get(crop_key, (default_rates, default_yield))

st.markdown(
f"""
<div style="color:#00ff00; font-weight:bold;">
Recommended crop: {pred}<br>
Estimated market rate: {rate:.1f} ₹/kg<br>
Estimated yield: {yld:.0f} kg/ha
</div>
""",
unsafe_allow_html=True,
)

# -------------------------------------------------
# 7) Basic statistics
# -------------------------------------------------
st.subheader("Basic Dataset Statistics")
st.write(data.describe())
