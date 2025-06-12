import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("rfr_model.pkl")

st.title("Medical insurance cost estimator")

# Input fields
age = st.number_input("Age",min_value=18,max_value=65,value=20,step =1)
sex = st.selectbox("Sex",["male","female"])
bmi = st.number_input("Bmi",min_value=15.0,max_value=45.0,value=20.0,step=0.1)
childern =st.selectbox("Children",[0,1,2,3,4,5])
smoker = st.selectbox("Smoker",["yes","no"])
region = st.selectbox("Region",["southwest","southeast","northwest","northeast"])

#input dataframe
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [childern],
    "smoker": [smoker],
    "region": [region]
})


input_df['bmi_smoker'] = input_df['bmi'] * (input_df['smoker'] == 'yes').astype(int)
input_df['age_smoker'] = input_df['age'] * (input_df['smoker'] == 'yes').astype(int)
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("predict Charges"):
    pred = model.predict(input_encoded)[0]
    st.success(f"Estimated Charges: ${pred:,.2f}")
