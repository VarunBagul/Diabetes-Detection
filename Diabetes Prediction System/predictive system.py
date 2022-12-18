# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model=pickle.load(open('D:\ML Projects Practice/Diabetes Prediction System/trained_model.sav','rb'))

#copy paste and replace classifier by loaded-model
input_data=(8,183,64,0,0,23.3,0.672,32)
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
  print("The person is not diabetic")
else:
    print("The person is diabetic")