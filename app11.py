import streamlit as st
import pickle
import pandas as pd

# Load the trained Lasso regression model
filename = 'lasso_regression_model.sav'
with open(filename, 'rb') as f:
    model = pickle.load(f)

# Create the Streamlit app
st.title('Monthly Revenue Prediction App')

# Input features
st.header('Enter Customer Information:')
total_orders = st.number_input('Total Orders', min_value=0)
avg_order_value = st.number_input('Average Order Value', min_value=0.0)
customer_lifetime_value = st.number_input('Customer Lifetime Value', min_value=0.0)
average_order_frequency = st.number_input('Average Order Frequency', min_value=0.0)

# Create a DataFrame with the user input
input_data = pd.DataFrame({
    'total_orders': [total_orders],
    'avg_order_value': [avg_order_value],
    'customer_lifetime_value': [customer_lifetime_value],
    'average_order_frequency': [average_order_frequency]
})

# Make a prediction when the user clicks the button
if st.button('Predict Monthly Revenue'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Monthly Revenue: ${prediction:.2f}')
