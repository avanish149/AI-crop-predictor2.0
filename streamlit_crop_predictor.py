import random
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# 1) Load data
# -------------------------------------------------
st.title("AI Crop Predictor Dashboard")
st.write("Upload your dataset, view the data, and try out crop predictions interactively!")

DATAFILE = "crop_recommendation.csv"
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

# Require economic columns to exist in the CSV
required_columns = {
    "N", "P", "K", "temperature", "humidity",
    "ph", "rainfall", "label", "rates", "yield",
}
missing = required_columns - set(data.columns)
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

# -------------------------------------------------
# 2) Train model in memory (no pickle)
# -------------------------------------------------
X = data.drop("label", axis=1)   # N,P,K,temperature,humidity,ph,rainfall,rates,yield
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.subheader("Model status")
st.write(f"RandomForest trained in app. Test accuracy: **{accuracy*100:.2f}%**")

# -------------------------------------------------
# 3) Dataset preview
# -------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(100))

# -------------------------------------------------
# 4) User input section
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

# Fixed defaults for economic features used at prediction time
default_rates = 25.5
default_yield = 3850

# -------------------------------------------------
# 5) Prediction
# -------------------------------------------------
if submit:
    # Order must match X.columns used during training
    input_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall, default_rates, default_yield]],
        columns=["N", "P", "K", "temperature", "humidity",
                 "ph", "rainfall", "rates", "yield"],
    )
    pred = model.predict(input_df)[0]
    st.success(f"Recommended Crop: {pred}")

# -------------------------------------------------
# 6) Basic statistics
# -------------------------------------------------
st.subheader("Basic Dataset Statistics")
st.write(data.describe())



