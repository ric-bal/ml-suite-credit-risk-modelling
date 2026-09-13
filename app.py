import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.set_page_config(layout='wide')
cols = st.columns([2,3], border=True, gap='medium')#, width=100000000)#, gap='xlarge')

if 'pred' not in st.session_state:
    st.session_state.pred = None
    st.session_state.input_df = {}

with cols[0]:
    encoders = {col: joblib.load(f'encoders/{col}_encoder.pkl') for col in ['sex', 'housing', 'saving_accounts', 'checking_account']}

    st.title('Credit Risk Prediction App')
    st.write('Enter applicant information to predict if the credit risk is good or bad')

    model_paths = list(Path('models/').glob('*.pkl'))
    model_dict = {}
    for p in model_paths:
        model_dict[p.name.replace('.pkl', '').replace('_', ' ')] = p

    

    col1, col2= st.columns([3,2])
    with col1:
        model_name = st.selectbox('Current Model', list(model_dict.keys()))
    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('Predict Risk', type='primary'):  
            st.session_state.pred = joblib.load(model_dict[model_name]).predict(st.session_state.input_df)[0]


    st.divider()


    age = st.number_input('Age', min_value=18, max_value=80, value=30)
    sex = st.selectbox('Sex', ['male', 'female'])
    job = st.number_input('Job (0-3)', min_value=0, max_value=3, value=1)
    housing = st.selectbox('Housing', ['own', 'rent', 'free'])
    saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'rich', 'quite rich'])
    checking_account = st.selectbox('Checking Account', ['little', 'moderate', 'rich'])
    credit_amount = st.number_input('Credit Amount', min_value=0, value=100)
    duration = st.number_input('Duration (months)', min_value=1, value=12)

    st.session_state.input_df = pd.DataFrame({
        'age': [age],
        'sex': [encoders['sex'].transform([sex])[0]],
        'job': [job],
        'housing': [encoders['housing'].transform([housing])[0]],
        'saving_accounts': [encoders['saving_accounts'].transform([saving_accounts])[0]],
        'checking_account': [encoders['checking_account'].transform([checking_account])[0]],
        'credit_amount': [credit_amount],
        'duration': [duration]
    })


with cols[1]:
    tab1, tab2 = st.tabs([
        'Results',
        'Data'
    ])

    with tab1:
        if st.session_state.pred:
            st.success(f'Predicted Credit Risk: GOOD')
        elif st.session_state.pred != None:
            st.error(f'Predicted Credit Risk: BAD')

    with tab2:
        st.header('qwe34')



""" 
TODO:
 - split screen model and result information
 - more models

"""
