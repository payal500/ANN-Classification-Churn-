import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#load train model

Model=tf.keras.models.load_model('Model.h5')

#load encoder and scaler
with open('gender.pkl','rb') as file:
    genders=pickle.load(file)
with open('geography.pkl','rb') as file:
    geographys=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scalers=pickle.load(file)

#build app
st.title("Customer Churn Prediction")

#user input
geography=st.selectbox('Geography',geographys.categories_[0])
gender=st.selectbox('Gender',genders.classes_)
age=st.slider('Age', 18 ,92)
balance=st.number_input('Balance')
CreditScore=st.number_input('CreditScore')
EstimatedSalary=st.number_input('EstimatedSalary')
Tenure=st.slider('Tenure',0,10)
NumOfProducts=st.slider('NumOfProducts',1,4)
HasCrCard=st.selectbox('HasCrCard',[0,1])
IsActiveMember=st.selectbox('IsActiveMember',[0,1])

#prepare the input data
input_data=pd.DataFrame({
     'CreditScore':[CreditScore],
     'Gender':[genders.transform([gender])[0]],
     'Age':[age],
     'Tenure':[Tenure],
     'Balance':[balance],
     'NumOfProducts':[NumOfProducts],
     'HasCrCard':[HasCrCard],
     'IsActiveMember':[IsActiveMember],
     'EstimatedSalary':[EstimatedSalary],
})

geo_encoder=geographys.transform([[geography]]).toarray()
geo_code_df=pd.DataFrame(geo_encoder,columns=geographys.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_code_df],axis=1)

#scale  the input data
input_scaled_data=scalers.transform(input_data)

#prediction 
predict=Model.predict(input_scaled_data)
prediction=predict[0][0]
st.write("Prediction of churn",prediction)
if prediction> 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn")