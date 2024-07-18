# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pandas as pd

# Load the data and preprocess as before
file_path = 'C:\\Users\\murruela2\\OneDrive - Cementos Progreso, S.A\\Desktop\\Git\\Git\\datos_macro.xlsx'
data = pd.read_excel(file_path)
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
data.set_index('date', inplace=True)
data.drop(columns=['Unnamed: 0', 'year', 'month'], inplace=True)

series = data['IMAEOriginal'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series.reshape(-1, 1))

def create_dataset(series, time_step=1):
    dataX, dataY = [], []
    for i in range(len(series) - time_step - 1):
        a = series[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(series[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 3
X, y = create_dataset(scaled_series, time_step)
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=1, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

plt.figure(figsize=(15, 6))
plt.plot(scaler.inverse_transform(scaled_series), label='Actual Data')
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='Training Predictions')
plt.plot(np.arange(len(train_predict) + (time_step * 2) + 1, len(scaled_series) - 1), test_predict, label='Testing Predictions')
plt.legend()
plt.title('IMAEOriginal Forecasting with LSTM')
plt.show()

