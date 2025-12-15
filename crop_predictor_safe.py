import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import pickle

# -----------------------------------------
# 1) Config
# -----------------------------------------
DATAFILE = "crop_recommendation.csv"
MODELFILE = "crop_model.pkl"

# EXACT features that Streamlit uses
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    # -------------------------------------
    # 2) Load data
    # -------------------------------------
    if not os.path.isfile(DATAFILE):
        print(f"ERROR: '{DATAFILE}' not found in current folder.")
        sys.exit(1)

    data = pd.read_csv(DATAFILE)
    print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    # Standardise column names
    column_map = {
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Potassium": "K",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "pH_Value": "ph",
        "Rainfall": "rainfall",
        "Crop": "label",
        "Label": "label",
    }
    data = data.rename(columns=column_map)

    # Check required columns
    required = set(FEATURE_COLS + ["label"])
    missing = required - set(data.columns)
    if missing:
        print(f"ERROR: dataset missing columns: {missing}")
        sys.exit(1)

    # -------------------------------------
    # 3) Build X, y (7 features only)
    # -------------------------------------
    X = data[FEATURE_COLS]
    y = data["label"]

    print("\nUsing features:", FEATURE_COLS)
    print("Classes:", sorted(y.unique()))
    print("X shape:", X.shape)

    # -------------------------------------
    # 4) Train / test split
    # -------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------------
    # 5) Train RandomForest
    # -------------------------------------
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # -------------------------------------
    # 6) Evaluation
    # -------------------------------------
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}\n")

    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred_test))

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred_test))

    # -------------------------------------
    # 7) Save model
    # -------------------------------------
    with open(MODELFILE, "wb") as f:
        pickle.dump(clf, f)

    print(f"\nSaved model to: {MODELFILE}")
    print("Model feature_names_in_:", clf.feature_names_in_)


if __name__ == "__main__":
    main()




