from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
import joblib
import streamlit as st

def make_model():
    input = keras.layers.Input(shape=(15,), name='Input')
    x = keras.layers.Dense(30, activation='relu', name='FC1')(input)
    x = keras.layers.Dropout(0.5, name='DO1')(x)
    x = keras.layers.Dense(16, activation='relu', name='FC2')(x)
    x = keras.layers.Dropout(0.5, name='DO2')(x)
    x = keras.layers.Dense(18, activation='relu', name='FC3')(x)
    x = keras.layers.Dropout(0.5, name='DO3')(x)
    output = keras.layers.Dense(1, activation='sigmoid', name='Output')(x)

    model = keras.models.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["Accuracy"])

    return model

# Title
st.header("ATAA Streamlit Machine Learning App")

# Input bar 1
Age = st.number_input("Enter Age")

# Dropdown input 2
Sex = st.selectbox("Select Sex", ("Female", "Male"))

# Dropdown input 3
Smoking = st.selectbox("Select Smoking", ("Yes", "No"))

# Input bar 4
BMI = st.number_input("Enter BMI")

# Input bar 5
BSA = st.number_input("Enter BSA")

# Input bar 6
Waist = st.number_input("Enter Waist")

# Input bar 7
SBP = st.number_input("Enter SBP")

# Input bar 8
DBP = st.number_input("Enter DBP")

# Input bar 9
HR = st.number_input("Enter HR")\

# Dropdown input 10
Dyslipidemia = st.selectbox("Select Dyslipidemia", ("Yes", "No"))

# Dropdown input 11
Hypertension = st.selectbox("Select Hypertension", ("Yes", "No"))

# Input bar 12
Creatinine = st.number_input("Enter Creatinine")

# Input bar 13
Glucose = st.number_input("Enter Glucose")

# Input bar 14
CRP = st.number_input("Enter CRP")

# Input bar 15
LDL = st.number_input("Enter LDL")

# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("clf.pkl")

    # Store inputs into dataframe

    X = pd.DataFrame([[Age, Sex, Smoking, BMI, BSA, Waist, SBP, DBP, HR, Dyslipidemia, Hypertension,
            Creatinine, Glucose, CRP, LDL]],
                     columns=["Age", "Sex", "Smoking", "BMI", "BSA", "Waist", "SBP", "DBP", "HR", "Dyslipidemia", "Hypertension",
            "Creatinine", "Glucose", "CRP", "LDL"])
    X = X.replace(["Female", "Male", "Yes", "No"], [1, 2, 1, 0])

    # Get prediction

    file_name = 'scaler_01.pkl'
    scaler = joblib.load(file_name)
    test_features = scaler.transform(X)
    test_features = np.clip(test_features, -5, 5)

    smote_model = make_model()
    smote_model.load_weights('smote_best_model.h5')
    BATCH_SIZE = 128
    test_predictions_smote = smote_model.predict(test_features, batch_size=BATCH_SIZE)

    prediction = test_predictions_smote[0][0]

    # Output prediction
    st.text(f"This instance is a {prediction}")