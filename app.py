import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load the trained RNN model
model = load_model("rnn_model.h5")  # Make sure you also upload this file!

# --- 2. Load all stock data at the start
folder_path = 'stock_data_folder'  # relative path inside the app folder
file_list = os.listdir(folder_path)

df_list = []
stock_names = []

for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        stock_name = file.split('.')[0]  # Example: 'ADANIGREEN.NS'
        df['Stock'] = stock_name
        df_list.append(df)
        stock_names.append(stock_name)

# Combine all stock data
all_stocks_data = pd.concat(df_list, ignore_index=True)

# --- 3. Streamlit App Layout
st.title("📈 Stock Price Prediction App")

# Dropdown to select stock
selected_stock = st.selectbox("Select a Company", sorted(stock_names))

# Filter the selected stock data
stock_data = all_stocks_data[all_stocks_data['Stock'] == selected_stock]

# Display the stock data
st.write(f"### Showing data for {selected_stock}", stock_data.tail(10))

# --- 4. Prepare the data for prediction
if 'Close' not in stock_data.columns:
    st.error("The selected stock data does not have 'Close' column.")
else:
    # Ensure 'Close' is numeric
    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
    stock_data = stock_data.dropna(subset=['Close'])

    if len(stock_data) < 60:
        st.error("Not enough data to make predictions. Need at least 60 data points.")
    else:
        close_data = stock_data['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        prediction_days = 60
        x_input = []
        for i in range(prediction_days, len(scaled_data)):
            x_input.append(scaled_data[i-prediction_days:i, 0])

        x_input = np.array(x_input)
        x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

        # Make predictions
        predictions = model.predict(x_input)
        predictions = scaler.inverse_transform(predictions)

        # --- 5. Plotting
        st.write("### Predicted vs Actual Prices")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data['Close'].iloc[prediction_days:].values, color='blue', label='Actual Prices')
        ax.plot(predictions, color='red', label='Predicted Prices')
        ax.set_title(f"Stock Price Prediction for {selected_stock}")
        ax.set_xlabel("Time (in second)")
        ax.set_ylabel("Price (in rupees)")
        ax.legend()
        st.pyplot(fig)

        st.write("### Predicted Prices Table")
        st.dataframe(predictions)
