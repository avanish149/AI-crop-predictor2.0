import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Check Python version
if sys.version_info < (3, 8):
    print("Warning: Python 3.8+ is recommended. Current:", sys.version)
    print("Running with Python:", sys.version.split()[0])

# Locate dataset
DATAFILE = "crop_recommendation.csv"
print(f"\nLooking for dataset file: {DATAFILE}")
if not os.path.isfile(DATAFILE):
    print(f"ERROR: Dataset file '{DATAFILE}' not found in current working directory:\n {os.getcwd()}")
    sys.exit(1)

# Load dataset
try:
    data = pd.read_csv(DATAFILE)
    print("Dataset loaded. Shape:", data.shape)
except Exception as e:
    print("ERROR reading CSV file. Exception:")
    raise

# Check & fix columns
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

# NOW we also require rates and yield to be present
required_columns = {
    "N", "P", "K", "temperature", "humidity",
    "ph", "rainfall", "label", "rates", "yield"
}
missing = required_columns - set(data.columns)
if missing:
    print(f"ERROR: Dataset is missing columns: {missing}")
    sys.exit(1)

print("Column check passed. Using these columns:")
print(list(data.columns))

# Prepare features and target (all columns except label)
X = data.drop("label", axis=1)   # includes N,P,K,temp,humidity,ph,rainfall,rates,yield
y = data["label"]
print("\nFeature dtypes:")
print(X.dtypes)

# Train/test split and model training
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTraining RandomForestClassifier... (this can take a few seconds)")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model trained. Accuracy on test set: {acc*100:.2f}%")
except Exception as e:
    print("ERROR during training or evaluation. Exception:")
    raise

# Save model
outname = "crop_model.pkl"
try:
    with open(outname, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {outname}")
except Exception as e:
    print("ERROR saving model:")
    raise

# Demo prediction – now includes rates and yield
def predict_crop(N, P, K, temperature, humidity, ph, rainfall,
                 rates, crop_yield):
    df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall, rates, crop_yield]],
        columns=["N", "P", "K", "temperature", "humidity",
                 "ph", "rainfall", "rates", "yield"]
    )
    return model.predict(df)[0]

print("\nDemo prediction (example inputs):")
try:
    demo = predict_crop(42, 0, 0, 25, 80, 6.5, 200, 25.5, 3850)
    print("Predicted crop:", demo)
except Exception as e:
    print("Prediction failed:")
    raise

print("\nAll done.")
