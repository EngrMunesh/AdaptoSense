"""
AdaptoSense – ANN-Based Sensor Fault Detection and Data Prediction
Author: Munesh Meghwar
Description: A neural network model for predicting missing sensor values using IoT data (Arduino + Python)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load and Preprocess Data
# ----------------------------

file_paths = {
    "humidity_1": "C3PO-humidity1.csv",
    "co2": "C3PO-carbon.csv",
    "photoresistor": "C3PO-photoresistor.csv",
    "temperature_1": "C3PO-temperature1.csv",
    "temperature_2": "C3PO-temperature_2.csv",
    "humidity_2": "YodaThings-yodahumidity.csv",
    "temperature_3": "YodaThings-Yodatemperature.csv",
    "yoda_light": "YodaThings-yodalight.csv"
}

data_frames = {key: pd.read_csv(path) for key, path in file_paths.items()}
min_len = min(len(df) for df in data_frames.values())

def normalize(series):
    return series / (2 * np.max(series))

scaled_data = {
    key: normalize(df.iloc[-min_len:, 1].astype(float))
    for key, df in data_frames.items()
}

X_full = np.column_stack([scaled_data[key] for key in [
    "humidity_1", "photoresistor", "temperature_1", "temperature_2",
    "humidity_2", "temperature_3", "co2", "yoda_light"
]])

# Prepare features and target
y = X_full[:, 1]
X = np.delete(X_full, 1, axis=1)

non_zero_mask = X_full[:, -1] != 0
filtered_data = X_full[non_zero_mask]
validation_data = X_full[~non_zero_mask]

X_train, X_test = train_test_split(filtered_data, test_size=0.1, random_state=42)

# ----------------------------
# 2. Define Neural Network
# ----------------------------

class SimpleANN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        self.w_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        self.activation = lambda x: scipy.special.expit(x)

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        hidden_outputs = self.activation(np.dot(self.w_ih, inputs))
        final_outputs = self.activation(np.dot(self.w_ho, hidden_outputs))
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        self.w_ho += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.w_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)
        return output_errors

    def predict(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hidden_outputs = self.activation(np.dot(self.w_ih, inputs))
        final_outputs = self.activation(np.dot(self.w_ho, hidden_outputs))
        return final_outputs

# ----------------------------
# 3. Train the ANN Model
# ----------------------------

ann = SimpleANN(input_nodes=7, hidden_nodes=70, output_nodes=1, learning_rate=0.01)
epochs = 100
rms_errors = []

for _ in range(epochs):
    epoch_errors = []
    for record in X_train:
        features = record[0:7]
        target = record[7:]
        error = ann.train(features, target)
        epoch_errors.append(error ** 2)
    mean_error = np.mean([e[0, 0] for e in epoch_errors])
    rms_errors.append(math.sqrt(mean_error))

# ----------------------------
# 4. Model Evaluation
# ----------------------------

predictions, targets, error_sq, target_sq = [], [], [], []

for record in X_test[:100]:
    features = record[0:7]
    target = record[7:]
    prediction = ann.predict(features)
    predictions.append(prediction)
    targets.append(target)
    error_sq.append((target - prediction) ** 2)
    target_sq.append(target ** 2)

rmse = math.sqrt(np.sum(error_sq)) / math.sqrt(np.sum(target_sq))
ce = 1 - rmse ** 2

print("Relative RMSE:", rmse)
print("Model Coefficient of Efficiency (C.E.):", ce)

# ----------------------------
# 5. Plot Results
# ----------------------------

plt.figure(figsize=(8, 4))
plt.plot(rms_errors, marker='o')
plt.title("RMS Error vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("RMS Error")
plt.grid(True)

plt.figure(figsize=(6, 6))
plt.scatter(predictions, targets, color='blue')
plt.title("Model Predictions vs Targets")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.plot(np.array(predictions).flatten(), label='Predictions', marker='o', linestyle='--')
plt.plot(np.array(targets).flatten(), label='Actual', marker='x', linestyle=':')
plt.title("Predictions vs Actuals")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
