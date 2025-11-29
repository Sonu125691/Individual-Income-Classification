import streamlit as st
import pickle
import pandas as pd
import numpy as np
import base64


with open("final_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("final_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("final_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":

    st.sidebar.title("About this App")
    st.sidebar.info("""
    This application predicts whether an individual’s annual income is above or below $50K.  
    The prediction is based on personal and demographic factors such as age, workclass, education, marital status, 
    occupation, relationship, capital gain, capital loss, weekly working hours, and native country.  

    The dataset used in this project is the well-known UCI Adult Income dataset, which contains 48,842 records 
    with 14 descriptive attributes and one target variable, *Income*. The target has two classes: "<=50K" and ">50K".  

    In this study, multiple machine learning models were tested, including Logistic Regression, Support Vector Classifier, 
    Decision Tree, Random Forest, Gaussian Naive Bayes, and XGBoost. After evaluation, the XGBoost Classifier achieved 
    the highest accuracy of **86.66%**,and it was selected as the final model for making predictions in this app.  
    """)
  
    st.title("Individual's Income Prediction")
    
    age = st.slider("Select Age", 0, 100, 25)

    workclass = st.selectbox("Your Work Class", ['Private', 'Local government Employ', 'Self-emp-not-inc', 'Federal-gov',
       'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    
    education = st.selectbox("Your Education", ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th',
       'Prof-school', '7th-8th', 'Bachelors', 'Masters', 'Doctorate',
       '5th-6th', 'Assoc-voc', '9th', '12th', '1st-4th', 'Preschool'] )
    
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced',
       'Separated', 'Married-spouse-absent', 'Married-AF-spouse'] )
    
    occupation = st.selectbox("Your Occupation", ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv',
       'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical',
       'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv',
       'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'])
    
    relationship = st.selectbox("Relationship", ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife',
       'Other-relative'])

    capital_gain = st.number_input("Capital Gain", 0, 1000000, 1000)

    capital_loss = st.number_input("Capital Loss", 0, 1000000, 0)

    hours_per_week = st.slider("Hours Per Week", 1, 100, 40)

    native_country = st.selectbox("Native Country", ['United-States', 'Peru', 'Guatemala', 'Mexico',
       'Dominican-Republic', 'Ireland', 'Germany', 'Philippines',
       'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam',
       'South', 'Columbia', 'Japan', 'India', 'Cambodia', 'Poland',
       'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal',
       'China', 'Nicaragua', 'Honduras', 'Iran', 'Scotland', 'Jamaica',
       'Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece',
       'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'France',
       'Holand-Netherlands'])
    

    user_inputs = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "capital-gain": capital_gain,	
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    user_df = pd.DataFrame([user_inputs])
    
    if st.button("Predict"):

        obj_columns = ["workclass","education","marital-status","occupation","relationship","native-country"]
        encoded = encoder.transform(user_df[obj_columns])
        encoded = pd.DataFrame(encoded, columns = encoder.get_feature_names_out(obj_columns))

        user_df = user_df.drop(columns = obj_columns)
        final = pd.concat([user_df, encoded], axis = 1)

        final = scaler.transform(final)

        prediction = model.predict(final)

        if prediction[0] == 1:
            st.info("Income is More than 50K$")
        else:
            st.info("Income is lesser than or equals to 50K$")

