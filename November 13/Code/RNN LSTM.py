import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_sequence(n_steps):
    x = np.array([i / 10 for i in range(n_steps)])
    y = np.sin(x)
    return x, y

n_steps = 100
x, y = generate_sequence(n_steps)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()

n_input = 10
X, Y = [], []
for i in range(len(y) - n_input):
    X.append(y[i:i + n_input])
    Y.append(y[i + n_input])

X = np.array(X)
Y = np.array(Y)

X = X.reshape((X.shape[0], X.shape[1], 1))

def build_and_train_rnn_model(X, Y, model_type='rnn'):
    model = Sequential()

    if model_type == 'rnn':
        model.add(SimpleRNN(50, activation='tanh', input_shape=(n_input, 1)))
    elif model_type == 'lstm':
        model.add(LSTM(50, activation='tanh', input_shape=(n_input, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(X, Y, epochs=20, verbose= 1)

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'])
    plt.title(f'Training Loss for {model_type.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    return model

print('\nTraining LSTM model:')
rnn_model = build_and_train_rnn_model(X, Y, model_type='rnn')

print('\nTraining LSTM model:')
lstm_model = build_and_train_rnn_model(X, Y, model_type='lstm')

def make_predictions(model, Y, model_type):
    x_input = Y[-n_input:].reshape(1, n_input, 1)
    prediction = model.predict(x_input, verbose=0)
    return prediction[0][0]

rnn_prediction = make_predictions(rnn_model, Y, 'RNN')
lstm_prediction = make_predictions(lstm_model, Y, 'LSTM')

print(f'\nRNN Next Value Prediction: {rnn_prediction:.4f}')
print(f'\nRNN Next Value Prediction: {lstm_prediction:.4f}')
