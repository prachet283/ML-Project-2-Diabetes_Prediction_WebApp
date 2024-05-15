# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:18:31 2024

@author: prachet
"""

import numpy as np
import pickle
import streamlit as st

#loading. the saved model
loaded_model = pickle.load(open("diabetes_trained_model.sav",'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    if(prediction[0]==0):
      return 'The person is not diabetic' 
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting input data from user
    
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure volume")
    SkinThickness = st.text_input("SkinThickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
    Age = st.text_input("Age of the person")
    
    # code for prediction
    diagnosis = ''
    
    #creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
