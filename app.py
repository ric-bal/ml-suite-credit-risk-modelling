import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import io
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#sns.set_style('darkgrid')
# plt.rcParams.update({
#     'savefig.facecolor': '#0E1117',
#     'figure.facecolor': '#0E1117',
#     'axes.facecolor': '#161B22',
#     'axes.edgecolor': '#30363D',
#     'axes.labelcolor': '#E6EDF3',
#     'text.color': '#E6EDF3',
#     'xtick.color': '#8B949E',
#     'ytick.color': '#8B949E',
#     'grid.color': '#30363D',
#     'grid.alpha': 0.5,
#     'font.size': 10
# })

st.set_page_config(layout='wide')
cols = st.columns([2,3], border=True, gap='medium')

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
    tab1, tab2, tab3 = st.tabs([
        'Results',
        'Data Overview',
        'EDA'
    ], height='stretch', width='stretch')

    with tab1:
        if st.session_state.pred:
            st.success(f'Predicted Credit Risk: GOOD')
        elif st.session_state.pred != None:
            st.error(f'Predicted Credit Risk: BAD')

    with tab2:
        data = {name: joblib.load(f'data/{name}.pkl') for name in ['df_model', 'X_train', 'X_test', 'y_train', 'y_test']}
        df = data['df_model']

        st.header('Data Summary')
        st.text(f'Dataframe shape: {df.shape}')
        st.dataframe(df)
        
    
        st.subheader('Data Info Output')
        buffer = io.StringIO()
        df.info(buf=buffer, memory_usage=True, show_counts=False)
        st.code(buffer.getvalue())

        st.subheader('Data Describe Output')
        st.dataframe(df.describe(include='all').T)

    with tab3:
        st.header('Exploratory Data Analysis')

        # distribution of numerical features - bar charts
        st.subheader('Distribution of Numerical Features')
        cols = ['age', 'credit_amount', 'duration']
        col1, col2, col3 = st.columns(3, gap='xsmall')

        for container, feature in zip([col1, col2, col3], cols):
            fig = px.histogram(
                df,
                x=feature,
                nbins=10,
            )

            fig.update_layout(
                title=dict(
                    text=feature,
                    x=0.5,
                    xanchor='center',
                    y=0.98,
                    yanchor='top'
                ),
                height=350,
                margin=dict(l=10, r=10, t=50, b=10),

                paper_bgcolor="#14191F",
                plot_bgcolor='#161B22',
            )

            fig.update_traces(
                marker_line_color="#141415",
                marker_line_width=2
            )

            container.plotly_chart(
                fig,
                width='stretch'
            )

        # distribution of numerical features - boxplots
        cols = ['age', 'credit_amount', 'duration']
        col1, col2, col3 = st.columns(3, gap='xsmall')

        for container, feature in zip([col1, col2, col3], cols):
            fig = px.box(
                df,
                y=feature,
            )

            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=50, b=10),

                paper_bgcolor="#14191F",
                plot_bgcolor='#161B22',
            )

            fig.update_traces(
                marker_line_color="#141415",
                marker_line_width=2
            )

            container.plotly_chart(
                fig,
                width='stretch'
            )
        st.write('All right skewed, mode < median < mean. ```credit_amount``` is especially right skewed - many datapoints above Q3.')



""" 
TODO:
 - split screen model and result information
 - more models

"""
