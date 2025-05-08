#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and feature names
gb_model = joblib.load("gb.pkl")
scaler = joblib.load("scaler0.pkl")
feature_names = joblib.load("feature_names.pkl")

# Define state and area codes
all_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
all_area_codes = ['area_code_408', 'area_code_415', 'area_code_510']

# Streamlit UI Styling
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #141E30, #243B55);
            color: white;
        }
        .stApp {
            background: linear-gradient(to right, #141E30, #243B55);
        }
        .stSidebar {
            background-color: #1E2A38 !important;
        }
        .stButton>button {
            background-color: #ff4b4b; 
            color: white; 
            border-radius: 10px; 
            font-size: 18px;
        }
        .stSuccess {
            background-color: #28a745; 
            color: white; 
            padding: 10px; 
            border-radius: 10px;
        }
        .stError {
            background-color: #dc3545; 
            color: white; 
            padding: 10px; 
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Customer Churn Prediction App")
st.markdown("Use this app to predict whether a customer is likely to churn.")

st.sidebar.header("🔧 Input Customer Details")

def get_user_input():
    user_data = pd.DataFrame(columns=feature_names)
    user_data.loc[0] = 0  # Initialize with zeros

    with st.sidebar:
        st.subheader("📞 Customer Profile")
        user_data["account.length"] = st.number_input("Account Length", min_value=0, step=1, value=100)
        user_data["voice.messages"] = st.number_input("Voice Messages", min_value=0, step=1, value=10)
        user_data["customer.calls"] = st.number_input("Customer Service Calls", min_value=0, step=1, value=2)
        
        st.subheader("📶 Usage Details")
        user_data["intl.mins"] = st.number_input("International Minutes", min_value=0.0, step=0.1, value=10.0)
        user_data["intl.calls"] = st.number_input("International Calls", min_value=0, step=1, value=3)
        user_data["day.mins"] = st.number_input("Day Minutes", min_value=0.0, step=0.1, value=180.0)
        user_data["day.calls"] = st.number_input("Day Calls", min_value=0, step=1, value=100)
        user_data["eve.mins"] = st.number_input("Evening Minutes", min_value=0.0, step=0.1, value=200.0)
        user_data["eve.calls"] = st.number_input("Evening Calls", min_value=0, step=1, value=100)
        user_data["night.mins"] = st.number_input("Night Minutes", min_value=0.0, step=0.1, value=200.0)
        user_data["night.calls"] = st.number_input("Night Calls", min_value=0, step=1, value=100)
        
        st.subheader("📜 Subscription Details")
        voice_plan = st.selectbox("Voice Plan", ["No", "Yes"])
        intl_plan = st.selectbox("International Plan", ["No", "Yes"])
        state = st.selectbox("State", all_states)
        area_code = st.selectbox("Area Code", all_area_codes)
        
    user_data["voice.plan"] = 1 if voice_plan == "Yes" else 0
    user_data["intl.plan"] = 1 if intl_plan == "Yes" else 0
    
    if f"state_{state}" in user_data.columns:
        user_data[f"state_{state}"] = 1
    if f"area_code_{area_code}" in user_data.columns:
        user_data[f"area_code_{area_code}"] = 1
    
    return user_data

input_data = get_user_input()
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Standardize numeric columns
num_cols = ["account.length", "voice.messages", "intl.mins", "intl.calls", "day.mins", "day.calls", "eve.mins", "eve.calls", "night.mins", "night.calls", "customer.calls"]
input_data[num_cols] = scaler.transform(input_data[num_cols])

st.markdown("### 🔍 Prediction")
if st.button("🚀 Predict Churn"):
    prediction = gb_model.predict(input_data)[0]
    probability = gb_model.predict_proba(input_data)[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🛎️ Churn Prediction", value="Churn" if prediction == 1 else "No Churn")
    with col2:
        st.metric(label="🔢 Churn Probability", value=f"{probability:.2%}")
    
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ This customer is unlikely to churn. (Probability: {probability:.2%})")


# In[ ]:




