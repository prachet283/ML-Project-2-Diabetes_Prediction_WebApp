# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:18:31 2024

@author: prachet
"""

import json
import pickle
import streamlit as st
import pandas as pd

#loading. the saved model
with open("Updated/columns.pkl", 'rb') as f:
    all_features = pickle.load(f)
with open("Updated/scaler.pkl", 'rb') as f:
    scalers = pickle.load(f)
with open("Updated/best_features_svc.json", 'r') as file:
    best_features_svc = json.load(file)
with open("Updated/best_features_lr.json", 'r') as file:
    best_features_lr = json.load(file)
with open("Updated/best_features_rfc.json", 'r') as file:
    best_features_rfc = json.load(file)
with open("Updated/diabetes_disease_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc = pickle.load(f)
with open("Updated/diabetes_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr = pickle.load(f)
with open("Updated/diabetes_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc = pickle.load(f)

#creating a function for prediction

def diabetes_prediction(input_data):

    df = pd.DataFrame([input_data], columns=all_features)

    df[all_features] = scalers.transform(df[all_features])
    
    df_best_features_svc = df[best_features_svc]
    df_best_features_lr = df[best_features_lr]
    df_best_features_rfc = df[best_features_rfc]
    
    prediction1 = loaded_model_svc.predict(df_best_features_svc)
    prediction2 = loaded_model_lr.predict(df_best_features_lr)
    prediction3 = loaded_model_rfc.predict(df_best_features_rfc)
    
    return prediction1 , prediction2, prediction3
  
    
  
def main():
    
    #giving a title
    st.title('Diabetes Prediction using ML')
    
    #getting input data from user
    
    Pregnancies = st.number_input("Number of Pregnancies",format="%.6f")
    Glucose = st.number_input("Glucose Level",format="%.6f")
    BloodPressure = st.number_input("BloodPressure volume",format="%.6f")
    SkinThickness = st.number_input("SkinThickness value",format="%.6f")
    Insulin = st.number_input("Insulin level",format="%.6f")
    BMI = st.number_input("BMI value",format="%.6f")
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction value",format="%.6f")
    Age = st.number_input("Age of the person",format="%.6f")

    

    # code for prediction
    diabetes_diagnosis_svc = ''
    diabetes_diagnosis_lr = ''
    diabetes_diagnosis_rfc = ''
    
    diabetes_diagnosis_svc,diabetes_diagnosis_lr,diabetes_diagnosis_rfc = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    #creating a button for Prediction
    if st.button("Predict Diabetes"):
        if(diabetes_diagnosis_rfc[0]==0):
            prediction = 'The person is not diabetic' 
        else:
            prediction = 'The person is diabetic'
        st.write(f"Prediction: {prediction}")
    
    if st.checkbox("Show Advanced Options"):
        if st.button("Predict Diabetes with Random Forest Classifier"):
            if(diabetes_diagnosis_rfc[0]==0):
                prediction = 'The person is not diabetic' 
            else:
                prediction = 'The person is diabetic'
            st.write(f"Prediction: {prediction}")
        if st.button("Predict Diabetes with Logistic Regression Model"):
            if(diabetes_diagnosis_lr[0]==0):
                prediction = 'The person is not diabetic' 
            else:
                prediction = 'The person is diabetic'
            st.write(f"Prediction: {prediction}")
        if st.button("Predict Diabetes with Support Vector Classifier"):
            if(diabetes_diagnosis_svc[0]==0):
                prediction = 'The person is not diabetic' 
            else:
                prediction = 'The person is diabetic'
            st.write(f"Prediction: {prediction}")    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
