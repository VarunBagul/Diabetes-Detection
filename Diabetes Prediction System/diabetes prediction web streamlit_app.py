# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:16:28 2022

@author: LENOVO
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('D:/ML Projects Practice/Diabetes Prediction System/trained_model.sav','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    #copy paste and replace classifier by loaded-model
    #input_data=(8,183,64,0,0,23.3,0.672,32)
    #random data from dataset
    #changing ip data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instance(we trained on 768 ex and 8 cols)
    #we are just using one data pt and model expects 768 ips so reshape for 1
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    #we have standardize and now we are giving random data
    #sso stand it again
    # for streamlite std_data=scaler.transform(input_data_reshaped)
    #print(std_data)
    #prediction=classifier.predict(std_data)
    prediction=loaded_model.predict(input_data_reshaped)

    #as classifier has stored data
    print(prediction)

    if(prediction[0]==0):
      return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
def main():
    
    #giving a title 
    st.title('Diabetes Prediction Web App')
    
    #getting input data
    
    Pregnancies=st.text_input('no of pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure value')
    SkinThickness=st.text_input('Skin Thickness  value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI  value')
    DiabetesPedigreeFunction=st.text_input('Diabetes PedigreeFunction value')
    Age=st.text_input(' age of the person is')
    
    #code for prediction
    diagnosis=''

    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)

if __name__=='__main__':
    main()

    
