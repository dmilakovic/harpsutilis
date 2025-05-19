#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:22:07 2025

@author: dmilakov
"""

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import numpy as np
from tensorflow import keras
from keras import layers
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter



def top_hat_2d(x, y, x0, y0, wx, wy):
    return np.where((np.abs(x - x0) <= wx / 2) & (np.abs(y - y0) <= wy / 2), 1, 0)

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def convolved_model(xy, x0, y0, sigma_x, sigma_y, wx, wy, a, b, c):
    x, y = xy
    dx, dy = x[1, 0] - x[0, 0], y[0, 1] - y[0, 0]  # Grid spacing
    
    top_hat = top_hat_2d(x, y, x0, y0, wx, wy)
    gaussian = gaussian_2d(x, y, x0, y0, sigma_x, sigma_y)
    
    gaussian /= np.sum(gaussian) * dx * dy
    convolved = convolve2d(top_hat, gaussian, mode='same', boundary='fill', fillvalue=0)
    
    convolved *= 1 / np.sum(convolved) * dx * dy
    
    background = a * x + b * y + c
    
    return convolved.ravel() + background.ravel()

def fast_approximation(data, sigma=1.0):
    return gaussian_filter(data, sigma=sigma)

def fit_convolved_model(x, y, data):
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    xy = (x_grid, y_grid)
    
    initial_guess = [np.mean(x), np.mean(y), 1.0, 1.0, 2.0, 2.0, 0, 0, np.mean(data)]
    popt, _ = curve_fit(convolved_model, xy, data.ravel(), p0=initial_guess)
    
    return popt

def build_neural_network(input_shape):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(6)  # Outputs: x0, y0, sigma_x, sigma_y, wx, wy
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_synthetic_data(size=1000, grid_size=(32, 32)):
    x = np.linspace(-5, 5, grid_size[0])
    y = np.linspace(-5, 5, grid_size[1])
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    data = []
    labels = []
    
    for _ in range(size):
        x0, y0 = np.random.uniform(-2, 2, 2)
        sigma_x, sigma_y = np.random.uniform(0.5, 2, 2)
        wx, wy = np.random.uniform(1, 3, 2)
        a, b, c = np.random.uniform(-0.1, 0.1, 3)
        synthetic_image = convolved_model((x_grid, y_grid), x0, y0, sigma_x, sigma_y, wx, wy, a, b, c).reshape(grid_size)
        
        data.append(synthetic_image)
        labels.append([x0, y0, sigma_x, sigma_y, wx, wy])
    
    return np.array(data)[..., np.newaxis], np.array(labels)

def train_neural_network(model, data, labels, epochs=10, batch_size=32):
    model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return model

def test_neural_network(model, test_size=100, grid_size=(32, 32)):
    test_data, test_labels = generate_synthetic_data(size=test_size, grid_size=grid_size)
    predictions = model.predict(test_data)
    
    mse = np.mean((predictions - test_labels) ** 2, axis=0)
    print("Mean Squared Error for each parameter:", mse)
    
    for i in range(5):
        print(f"Test {i+1} - True: {test_labels[i]}, Predicted: {predictions[i]}")
    
    return mse

if __name__=="__main__":
    # Build and train the model
    input_shape = (32, 32, 1)
    model = build_neural_network(input_shape)
    train_data, train_labels = generate_synthetic_data(size=1000)
    model = train_neural_network(model, train_data, train_labels, epochs=10)
    
    # Test the trained model
    test_neural_network(model)