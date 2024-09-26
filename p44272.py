import streamlit as st%%writefile app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso regression model
filename = 'lasso_regression_model.sav'
model = pickle.load(open(filename, 'rb'))

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
import pickle
import pandas as pd

# Load the trained model
filename = 'crop_recommendation_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a function for prediction
def crop_prediction(input_data):
    # Change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    return prediction[0]

def main():
    # Giving a title
    st.title('Crop Recommendation System')

    # Getting the input data from the user
    N = st.text_input("Nitrogen")
    P = st.text_input("Phosphorus")
    K = st.text_input("Potassium")
    temperature = st.text_input("Temperature")
    humidity = st.text_input("Humidity")
    ph = st.text_input("pH")
    rainfall = st.text_input("Rainfall")

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Crop Recommendation'):
        input_data = [N, P, K, temperature, humidity, ph, rainfall]
        diagnosis = crop_prediction(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()
