
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from lightgbm import LGBMClassifier
import joblib

st.set_page_config(layout= 'wide', page_title='Car Insurance Deployment')

st.image('https://thumbs.dreamstime.com/b/car-insurance-logo-auto-protection-safety-secure-shield-stock-photo-generative-ai-illustration-representing-vehicle-security-369131383.jpg?w=992')

df = pd.read_csv('cleaned_df.csv', index_col= 0)
# st.dataframe(df)

kidsdriv = st.selectbox('PLease provid number of driving kids', df['KIDSDRIV'].unique())
age = st.sidebar.slider('Enter your Age', 16, 81)
homekids = st.selectbox('PLease provid total number of kids', df['HOMEKIDS'].unique())
yoj = st.sidebar.slider('Please provide number of years on job', min_value= int(df['YOJ'].min()), max_value= int(df['YOJ'].max()), step= 1)
income = st.number_input('Please provide your income', min_value= df['INCOME'].min(), max_value= df['INCOME'].max())
parent1 = st.selectbox('PLease select whether you are single parent or not', df['PARENT1'].unique())
home_val = st.number_input('Please provide your home value', min_value= df['HOME_VAL'].min(), max_value= df['HOME_VAL'].max())
mstatus = st.sidebar.radio('Marital Status', df['MSTATUS'].unique())
gender = st.sidebar.radio('Gender', df['GENDER'].unique())
education = st.selectbox('Please enter your educational background', df['EDUCATION'].unique())
occupation = st.selectbox('Please enter your Occupation', df['OCCUPATION'].unique())
travtime = st.sidebar.slider('Please enter your travel time in minutes', min_value= df['TRAVTIME'].min(), max_value= df['TRAVTIME'].max(), step= 1)
car_use = st.selectbox('PLease provid your car usage', df['CAR_USE'].unique())
bluebook = st.number_input('Please provide your car value', min_value= df['BLUEBOOK'].min(), max_value= df['BLUEBOOK'].max())
tif = st.sidebar.slider('Loyalty years', min_value= df['TIF'].min(), max_value= df['TIF'].max(), step= 1)
car_type = st.selectbox('PLease provid your car type', df['CAR_TYPE'].unique())
red_car = st.selectbox('Red car or Not', df['RED_CAR'].unique())
oldclaim = st.number_input('Please provide your old claim amount', min_value= df['OLDCLAIM'].min(), max_value= df['OLDCLAIM'].max())
#clm_freq = st.sidebar.slider('Please provide number of previous claims', min_value= df['CLM_FREQ'].min(), max_value= df['CLM_FREQ'].max(), step= 1)
revoked = st.selectbox('License Revoked within 7 years', df['REVOKED'].unique())
mvr_pts = st.sidebar.slider('Please provide vechile record points', min_value= df['MVR_PTS'].min(), max_value= df['MVR_PTS'].max(), step= 1)
#clm_amt = st.number_input('Please provide your total claims amount', min_value= df['CLM_AMT'].min(), max_value= df['CLM_AMT'].max())
car_age = st.sidebar.slider('Please enter your car age', min_value= int(df['CAR_AGE'].min()), max_value= int(df['CAR_AGE'].max()), step= 1)
urbanicity = st.selectbox('Urbanicity', df['URBANICITY'].unique())
customer_loyalty = st.selectbox('Customer Loyalty', df['customer_loyalty'].unique())

# Import Model pkl file
model = joblib.load('lightgbm.pkl')

new_data = pd.DataFrame(columns= df.columns.drop('CLAIM_FLAG'), 
                        data= [[kidsdriv, age, homekids, yoj, income, parent1, home_val,
                            mstatus, gender, education, occupation, travtime, car_use,
                            bluebook, tif, car_type, red_car, oldclaim,
                            revoked, mvr_pts, car_age, urbanicity,
                            customer_loyalty]])

if st.button('Predict'):

    result = model.predict(new_data)[0]

    if result == 0:
        st.write('Desired Customer')

    else:
        st.write('Undesired Customer')
