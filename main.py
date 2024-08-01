import os
import pandas as pd
import requests
from io import StringIO

# Function to load data from GitHub
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        txt_data = StringIO(response.text)
        df = pd.read_csv(txt_data, sep=",")
        return df
    else:
        raise Exception(f"Failed to load data from {url}")

# URLs to the data files in the GitHub repository
etfs_url = 'https://raw.githubusercontent.com/rosswylie/4375FinalProject/main/archive/ETFs/aadr.us.txt'
stocks_url = 'https://raw.githubusercontent.com/rosswylie/4375FinalProject/main/archive/Stocks/aapl.us.txt'

# Load the data
df_etf = load_data_from_github(etfs_url)
df_stock = load_data_from_github(stocks_url)

# Display the data
print(df_stock.head())

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_stock['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 60  # Example sequence length
X, y = create_sequences(scaled_data, seq_length)

# Split into training and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, batch_size=1, epochs=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot the results
import matplotlib.pyplot as plt

train = df_stock[:split + seq_length]
valid = df_stock[split + seq_length:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()



