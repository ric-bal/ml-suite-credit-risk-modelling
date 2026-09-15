import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import io
import plotly.express as px

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score)


if 'pred' not in st.session_state:
    st.session_state.pred = None
    st.session_state.input_df = {}

data = {name: joblib.load(f'data/{name}.pkl') for name in ['df_og', 'df_model', 'X_train', 'X_test', 'y_train', 'y_test']}
encoders = {col: joblib.load(f'encoders/{col}_encoder.pkl') for col in ['sex', 'housing', 'saving_accounts', 'checking_account']}
models = {model.name.replace('_', ' ').replace('.pkl', ''): joblib.load(f'models/{model.name}') for model in list(Path('models/').glob('*.pkl'))}
current_model = None

st.set_page_config(layout='wide')
cols = st.columns([2,3], border=True, gap='medium')

with cols[0]:
    st.title('Credit Risk Prediction App')
    st.write('Enter applicant information to predict if the credit risk is good or bad')

    col1, col2= st.columns([3,2])
    with col1:
        model_name = st.selectbox('Current Model', list(models.keys()))
        st.session_state.pred = None
    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('Predict Risk', type='primary'):  
            current_model = models[model_name]
            st.session_state.pred = current_model.predict(st.session_state.input_df)[0]

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
    tab1, tab2, tab3, tab4 = st.tabs([
        'Results',
        'Data Overview',
        'Exploratory Data Analysis',
        'Feature Engineering'
    ])

    with tab1:
        st.header(f'Current Model - {model_name}')
        if st.session_state.pred:
            st.success(f'Predicted Credit Risk: GOOD')
        elif st.session_state.pred != None:
            st.error(f'Predicted Credit Risk: BAD')
        else:
            st.info('Predicted Credit Risk: ')

        scroll_container = st.container(height=740, border=True)
        with scroll_container:
            if st.session_state.pred != None:
                X_train = data['X_train']
                X_test = data['X_test']
                y_train = data['y_train']
                y_test = data['y_test']

                y_pred = current_model.predict(X_test)

                st.subheader('Results Summary')
                score_dict = {
                    'Accuracy': (accuracy_score(y_test, y_pred), 'Accuracy: the overall proportion of correct predictions (positive and negative) out of total predictions'),
                    'Precision': (precision_score(y_test, y_pred), 'Precision: the proportion of positive predictions that were correct (minimizes false positives)'),
                    'Recall': (recall_score(y_test, y_pred), 'Recall: the proportion of actual positive instances that the model managed to capture (minimizes false negatives)'),
                    'F1-score': (f1_score(y_test, y_pred), 'F1-score: the harmonic mean of precision and recall, providing a single balanced score')
                }
                for score in score_dict:
                    st.write(f'> {score_dict[score][1]}')
                    st.code(f'{score}: {score_dict[score][0]}')
                    
                st.subheader('Score Table')
                st.code(classification_report(y_test, y_pred))

                st.subheader('ROC-AUC Score')
                st.write('> How well the model is able to distinguish positive examples above negative examples across classification thresholds')
                y_prob = current_model.predict_proba(X_test)[:, 1]
                st.code(f'ROC-AUC: {roc_auc_score(y_test, y_prob)}')

                st.subheader('Model Parameter Settings')
                params_df = pd.DataFrame(list(current_model.get_params().items()), columns=["Parameter", "Value"])
                params_df['Value'] = params_df['Value'].astype('str')#apply(lambda x: round(x, 2)).astype('str')
                st.dataframe(params_df, width=500)

                st.subheader('Model Feature Importance')
                st.write('> Importance of each variable in determining the target within the fitted model')
                importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': current_model.feature_importances_}).sort_values("Importance", ascending=False)
                importance_df['Importance'] = importance_df['Importance'].apply(lambda x: round(x, 2)).astype('str')
                st.dataframe(importance_df, width=500)

    with tab2:
        df_model = data['df_model']

        st.header('Data Overview')
        st.text('(Post-EDA data)')
        st.text(f'Dataframe shape: {df_model.shape}')
        st.dataframe(df_model)
        
    
        st.subheader('Data Info Output')
        buffer = io.StringIO()
        df_model.info(buf=buffer, memory_usage=True, show_counts=False)
        st.code(buffer.getvalue())

        st.subheader('Data Describe Output')
        st.dataframe(df_model.describe(include='all').T)

    with tab3:
        st.header('Exploratory Data Analysis')
        st.text('Exploring the quality and features of the unprocessed data to determine how to prepare the dataset for general machine learning methods.')
        
        scroll_container = st.container(height=780, border=True)
        with scroll_container:

            df_og = data['df_og']
            if 'unnamed:_0' in df_og.columns:
                df_og.drop(columns='unnamed:_0', inplace=True)

            # original df
            st.subheader(f'Original dataframe shape: {df_og.shape}')
            st.dataframe(df_og)
            st.divider()

            # data info output
            st.subheader('Data Info Output')
            buffer = io.StringIO()
            df_og.info(buf=buffer, memory_usage=True, show_counts=False)
            st.code(buffer.getvalue())
            st.divider()

            # risk value counts
            st.subheader('Target Value Counts')
            st.code(df_og['risk'].value_counts())
            st.write('Heavy bias towards ```bad``` counts, will need to rebalance. Could try over/undersampling, class weighting, etc, but be careful with carrying bias throuugh generated samples with a relatively small dataset (1000 rows only).')
            st.divider()

            # dupes and nas 
            st.subheader('Duplicates and NaNs Check')
            st.code(f'df_og.isna().sum(): \n\n{df_og.isna().sum()}')
            st.code(f'df_og.duplicated().sum(): {df_og.duplicated().sum()}')
            st.write('No duplicated rows. ```saving_accounts``` and ```checking_account``` contaian many NaN values but I have a feeling these two are significant predictors. May be helpful to drop rows that do not contain values for these features.')

            df_og = df_og.dropna().reset_index(drop=True)
            st.code(f'df_og = df_og.dropna().reset_index(drop=True)')
            st.divider()
            
            # distribution of numerical features - bar charts
            st.subheader('Distribution of Numerical Features')
            num_cols = ['age', 'credit_amount', 'duration']
            col1, col2, col3 = st.columns(3, gap='xsmall')

            for container, feature in zip([col1, col2, col3], num_cols):
                fig = px.histogram(
                    df_og,
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

                    paper_bgcolor='#14191F',
                    plot_bgcolor='#161B22',
                )
                fig.update_traces(
                    marker_line_color='#141415',
                    marker_line_width=2
                )
                container.plotly_chart(
                    fig,
                    width='stretch'
                )

            # distribution of numerical features - boxplots
            col1, col2, col3 = st.columns(3, gap='xsmall')
            for container, feature in zip([col1, col2, col3], num_cols):
                fig = px.box(
                    df_og,
                    y=feature,
                )
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10),

                    paper_bgcolor='#14191F',
                    plot_bgcolor='#161B22',
                )
                fig.update_traces(
                    marker_line_color='#141415',
                    marker_line_width=2
                )
                container.plotly_chart(
                    fig,
                    width='stretch'
                )
            st.write('All right skewed, mode < median < mean. ```credit_amount``` is especially right skewed - many datapoints above Q3.')

            # distribution of categorical features - bar charts
            st.subheader('Distribution of Categorical Features')
            cat_cols = [['sex', 'job'], ['housing', 'saving_accounts'], ['checking_account', 'purpose']]
            col1, col2, col3 = st.columns(3, gap='xsmall')

            for container, features in zip([col1, col2, col3], cat_cols):
                for feature in features:
                    fig = px.histogram(
                        df_og,
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

                        paper_bgcolor='#14191F',
                        plot_bgcolor='#161B22',
                    )
                    fig.update_traces(
                        marker_line_color='#141415',
                        marker_line_width=2
                    )
                    container.plotly_chart(
                        fig,
                        width='stretch'
                    )

            # pearson correlation coeff
            st.subheader('Pearson Correlation Coefficient: Feature Collinearity')

            corr_cols = ['age', 'job', 'credit_amount', 'duration']
            corr = df_og[corr_cols].corr()
            corr = corr.iloc[::-1]

            fig = px.imshow(
                corr,
                text_auto='.2f',
                zmin=0,
                zmax=1,
                color_continuous_scale='teal'
                
            )
            scroll_container.plotly_chart(
                fig,
                width='stretch',
            )
            st.write('These features were likely to be collinear but instead appeared to be mostly independent. Watch out for ```duration``` and ```credit_amount```, relatively high PCC value.')
            st.divider()

            # pearson correlation coeff
            st.subheader('Pivot Table: Mean Credit Amount for each Housing Category, for each Purpose')
            pivot = pd.pivot_table(df_og, values='credit_amount', index='housing',columns='purpose')
            st.code(pivot)
            st.divider()

            # Credit Amount vs Age, coloured by Sex, sized by Duration
            st.subheader('Credit Amount vs Age, coloured by Sex, sized by Duration')

            fig = px.scatter(
                df_og,
                x='age',
                y='credit_amount',
                color='sex',
                size='duration',
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            scroll_container.plotly_chart(
                fig,
                width='stretch',
                height=600
            )
            st.write('Most data points grouped in low ```credit_amount``` and low ```age``` and tend to have low ```duration```. Highest ```duration``` also correlates with higher ```credit_amount``` and ```age```. Unable to tell ```gender``` correlation information from this plot.')
            st.divider()

            # Credit Amount distribution by Savings Accounts
            st.subheader('Credit Amount Distribution by Savings Accounts')

            fig = px.violin(
                df_og,
                x='saving_accounts',
                y='credit_amount'
            )
            scroll_container.plotly_chart(
                fig,
                width='stretch',
                height=600
            )
            st.write('```credit_amount``` is heavily right-skewed across all ```saving_accounts``` categories, most loans at low amounts. High value outliers across all categories. No clear relationship between categories.')
            st.divider()

            # target perc
            st.subheader('Target Percentages')
            st.code(f'df_og[\'risk\'].value_counts(normalize=True) * 100: \n\n{df_og['risk'].value_counts(normalize=True) * 100}')
            st.write('Dataset target has evened out and lowered the effect of potential dataset bias, at the cost of some data loss (dropped rows).')
            st.divider()

            # age, credit amount, and duration distribution by risk
            st.subheader('Distribution of Numerical Features by Risk')
            cols = ['age', 'credit_amount', 'duration']
            col1, col2, col3 = st.columns(3, gap='xsmall')

            for container, feature in zip([col1, col2, col3], cols):
                fig = px.box(
                    df_og,
                    x='risk',
                    y=feature,
                    color='risk',
                )
                fig.update_layout(
                    title=dict(
                        text=(f'\'{feature}\' by Risk'),
                        x=0.5,
                        xanchor='center',
                        y=0.98,
                        yanchor='top'
                    ),

                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10),

                    paper_bgcolor='#14191F',
                    plot_bgcolor='#161B22',
                )
                fig.update_traces(
                    marker_line_color='#141415',
                    marker_line_width=2
                )
                container.plotly_chart(
                    fig,
                    width='stretch'
                )
            st.divider()

            # Distribution of Categorical Features by Risk
            st.subheader('Distribution of Categorical Features by Risk')
            cat_cols = [['sex', 'job'], ['housing', 'saving_accounts'], ['checking_account', 'purpose']]
            col1, col2, col3 = st.columns(3, gap='xsmall')

            for container, features in zip([col1, col2, col3], cat_cols):
                for feature in features:
                    fig = px.histogram(
                        df_og,
                        x=feature,
                        color='risk',
                        nbins=10,
                        barmode='group'
                    )
                    fig.update_layout(
                        title=dict(
                            text=(f'\'{feature}\' by Risk'),
                            x=0.5,
                            xanchor='center',
                            y=0.98,
                            yanchor='top'
                        ),
                        height=350,
                        margin=dict(l=10, r=10, t=50, b=10),

                        paper_bgcolor='#14191F',
                        plot_bgcolor='#161B22',
                    )
                    fig.update_traces(
                        marker_line_color='#141415',
                        marker_line_width=2
                    )
                    container.plotly_chart(
                        fig,
                        width='stretch'
                    )
            st.divider()

        with tab4:
            st.header('Feature Engineering')
            st.code("Chosen Features:  ['age', 'sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'credit_amount', 'duration']")
            st.dataframe(df_model)

            st.subheader('Categorical Data Encoders')

            for col in encoders:
                st.code(f'{col} \nClasses: {encoders[col].classes_} \nEncoding: {encoders[col].transform(encoders[col].classes_)}')
                