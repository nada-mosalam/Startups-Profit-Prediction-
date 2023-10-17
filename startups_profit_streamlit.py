import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
lin_reg = joblib.load('lin_reg.pkl')

# Load the pipeline
full_pipeline = joblib.load('full_pipeline.pkl')

# Load the data
startups = pd.read_csv('50_Startups.csv')
startups.columns = startups.columns.str.lower()

# Create a title and sub-title
st.title("Startups Profit Prediction App")

st.write("""
This app predicts the profit of startups
""")

# Take the input from the user
# numerical attributes
r_and_d = st.slider('r&d spend', float(startups['r&d spend'].min()), float(startups['r&d spend'].max()))
administration = st.slider('administration', float(startups['administration'].min()), float(startups['administration'].max()))
marketing_spend	 = st.slider('marketing spend', float(startups['marketing spend'].min()), float(startups['marketing spend'].max()))

# categorical
state = st.selectbox('state', ('New York', 'California', 'Florida'))

# Store the variables in a dictionary
user_data = {'r&d spend': r_and_d,
'administration': administration,   
'marketing spend': marketing_spend,
'state': state}

# Transform the data into a data frame
features = pd.DataFrame(user_data, index=[0])

# Pipeline
features_prepared = full_pipeline.transform(features)

# Predict the output
prediction = lin_reg.predict(features_prepared)[0]

# Set a subheader and display the prediction
st.subheader('Profit Prediction:')
st.markdown('''# $ {} '''.format(round(prediction), 2))
