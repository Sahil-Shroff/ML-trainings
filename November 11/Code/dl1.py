# import tensorflow as tf
# print('tensorflow version: ', tf.__version__)

# import torch
# print('py torch: ', torch.__version__)

import numpy as np
import matplotlib.pyplot as plt

# x = np.random.random()

def generate_data(num_samples=100):
    np.random.seed(42)
    X = np.random.randn(num_samples, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)
    return X, y

X, y = generate_data()

def predict(x, weights, bias):
    return 1 if np.dot(x, weights) + bias >= 0 else -1

def perceptron(X, y, learning_rate=0.01, epochs=1000):
    weights= np.zeros(X.shape[1])
    bias = 0.2
    for epoch in range(epochs):
        for i, x in enumerate(X):
            update = learning_rate * (y[i] - predict(x, weights, bias))
            weights += update * x
            bias = update
    return weights, bias

weights, bias = perceptron(X, y)

def plot_decision_boundary(X, y, weights, bias):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    x_values = np.linspace(min(X[:, 1]), max(X[:, 0]), 100)
    y_values = -(weights[0] * x_values + bias) / weights[1]
    plt.plot(x_values, y_values, 'k--')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, weights, bias)