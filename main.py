import os
import pandas as pd
import requests
from io import StringIO

# Function to load data from GitHub
def fetch_data_from_github(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        data_as_string = StringIO(response.text)
        dataframe = pd.read_csv(data_as_string, sep=",")
        return dataframe
    else:
        raise Exception(f"Failed to load data from {file_url}")

# URLs to the data files in the GitHub repository
etf_data_url = 'https://raw.githubusercontent.com/rosswylie/4375FinalProject/main/archive/ETFs/aadr.us.txt'
stock_data_url = 'https://raw.githubusercontent.com/rosswylie/4375FinalProject/main/archive/Stocks/aapl.us.txt'

# Load the data
etf_dataframe = fetch_data_from_github(etf_data_url)
stock_dataframe = fetch_data_from_github(stock_data_url)

# Display the data
print(stock_dataframe.head())

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalize the data
data_scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = data_scaler.fit_transform(stock_dataframe['Close'].values.reshape(-1, 1))

# Create sequences
def generate_sequences(data_array, sequence_length):
    sequence_list = []
    label_list = []
    for index in range(len(data_array) - sequence_length):
        sequence_list.append(data_array[index:index+sequence_length])
        label_list.append(data_array[index+sequence_length])
    return np.array(sequence_list), np.array(label_list)

sequence_length = 60  # Example sequence length
sequences_X, labels_y = generate_sequences(normalized_data, sequence_length)

# Split into training and test sets
split_index = int(0.8 * len(sequences_X))
training_X, testing_X = sequences_X[:split_index], sequences_X[split_index:]
training_y, testing_y = labels_y[:split_index], labels_y[split_index:]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build and compile the model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.summary()

# Train the model
training_history = lstm_model.fit(training_X, training_y, batch_size=1, epochs=1)

# Evaluate the model
model_loss = lstm_model.evaluate(testing_X, testing_y)
print(f'Test Loss: {model_loss}')

# Make predictions
forecast_predictions = lstm_model.predict(testing_X)
forecast_predictions = data_scaler.inverse_transform(forecast_predictions)

# Plot the results
import matplotlib.pyplot as plt

train_data = stock_dataframe[:split_index + sequence_length]
validation_data = stock_dataframe[split_index + sequence_length:]
validation_data['Predicted'] = forecast_predictions

plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train_data['Close'])
plt.plot(validation_data[['Close', 'Predicted']])
plt.legend(['Train', 'Validation', 'Predicted'], loc='lower right')
plt.show()
