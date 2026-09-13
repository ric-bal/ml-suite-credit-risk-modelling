import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

encoders = {col: joblib.load(f'encoders/{col}_encoder.pkl') for col in ['sex', 'housing', 'saving_accounts', 'checking_account']}

st.title('Credit Risk Prediction App')
st.write('Enter applicant information to predict if the credit risk is good or bad')

model_paths = list(Path('models/').glob('*.pkl'))
model_dict = {}
for p in model_paths:
    model_dict[p.name.replace('.pkl', '').replace('_', ' ')] = p

model_name = st.selectbox('Current Model', list(model_dict.keys()))
st.divider()

age = st.number_input('Age', min_value=18, max_value=80, value=30)
sex = st.selectbox('Sex', ['male', 'female'])
job = st.number_input('Job (0-3)', min_value=0, max_value=3, value=1)
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox('Checking Account', ['little', 'moderate', 'rich'])
credit_amount = st.number_input('Credit Amount', min_value=0, value=100)
duration = st.number_input('Duration (months)', min_value=1, value=12)

input_df = pd.DataFrame({
    'age': [age],
    'sex': [encoders['sex'].transform([sex])[0]],
    'job': [job],
    'housing': [encoders['housing'].transform([housing])[0]],
    'saving_accounts': [encoders['saving_accounts'].transform([saving_accounts])[0]],
    'checking_account': [encoders['checking_account'].transform([checking_account])[0]],
    'credit_amount': [credit_amount],
    'duration': [duration]
})

if st.button('Predict Risk'):
    pred = joblib.load(model_dict[model_name]).predict(input_df)[0]

    if pred:
        st.success(f'Predicted Credit Risk: good')
    else:
        st.error(f'Predicted Credit Risk: bad')



""" 
TODO:
 - split screen model and result information
 - more models

"""
