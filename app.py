import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
load_model = tf.keras.models.load_model


# Load the trained model
model = load_model("rnn_model.h5")

# Set the page title
st.title("Stock Price Prediction App")

# File uploader for the user to upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Show the first few rows of the data
    st.write("### Uploaded Data", data.head())

    # Check if the necessary columns are available
    if 'Close' not in data.columns:
        st.error("The uploaded file does not contain the 'Close' column.")
    else:
        # Ensure the 'Close' column is numeric, removing any rows that have non-numeric data
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])  # Drop rows where 'Close' is NaN after coercion
        
        if len(data) < 60:
            st.error("Not enough data to make predictions. Please provide at least 60 data points.")
        else:
            # Extract the 'Close' column values
            close_data = data['Close'].values.reshape(-1, 1)

            # Scale the data using MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_data)

            # Prepare the data for prediction (using the last 60 days)
            prediction_days = 60
            x_input = []
            for i in range(prediction_days, len(scaled_data)):
                x_input.append(scaled_data[i-prediction_days:i, 0])

            # Convert the list into a numpy array
            x_input = np.array(x_input)
            x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

            # Make predictions
            predictions = model.predict(x_input)
            predictions = scaler.inverse_transform(predictions)

            # Plot the predictions and the actual data
            st.write("### Predicted vs Actual Prices")

            # Plot actual vs predicted values
            plt.figure(figsize=(12, 6))
            plt.plot(data['Close'].iloc[prediction_days:].values, color='blue', label='Actual Prices')
            plt.plot(predictions, color='red', label='Predicted Prices')
            plt.title("Stock Price Prediction")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot()

            # Show the prediction values
            st.write("### Predicted Stock Prices", predictions)

# Add a footer to the app
st.markdown("### Built with Streamlit and TensorFlow")
