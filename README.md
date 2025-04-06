# 🌐 AdaptoSense – ANN-Based Sensor Fault Detection and Data Prediction

**AdaptoSense** is an AI-driven system designed for **real-time sensor fault detection** and **data prediction** in smart building environments. Developed using **Python**, **Arduino Cloud**, and **deep learning**, the system ensures data integrity and continuous monitoring by predicting missing or faulty sensor values using an **Artificial Neural Network (ANN)**.

---

## 🧠 Project Summary

This project integrates an **Arduino-based IoT sensor network** with a **custom-built ANN model** to monitor and predict environmental parameters such as:

- **Temperature**
- **Humidity**
- **CO₂ concentration**
- **Ambient light intensity**

If any sensor fails or transmits zero values, the ANN estimates the missing data with high accuracy, ensuring uninterrupted environmental monitoring.

---

## ⚙️ System Overview

### ✅ Hardware Components
- Arduino MKR Wi-Fi 1010
- CO₂ Sensor (SCD41)
- Digital Temperature Sensor
- Photoresistor Sensor (Light)
- OPLA Weather Station
- Prototype Shield (for wireless communication)

### 💻 Software Stack
- **Arduino IDE** – Data acquisition and transmission
- **Arduino IoT Cloud** – Real-time sensor data logging
- **Python (Google Colab)** – ANN model training and prediction
- **Libraries**: `numpy`, `pandas`, `scipy`, `matplotlib`, `sklearn`

---

## 🧮 Neural Network Architecture

| Layer          | Configuration       |
|----------------|---------------------|
| Input Layer    | 7 nodes (features)  |
| Hidden Layer   | 70 nodes            |
| Output Layer   | 1 node (predicted)  |
| Activation     | Sigmoid             |
| Optimizer      | Gradient Descent    |
| Epochs         | 100                 |
| Loss Function  | Mean Squared Error  |

---

## 🔁 Workflow

1. **Sensor Data Collection**  
   Environmental data is collected every 5 seconds and stored in the Arduino IoT Cloud.

2. **Fault Simulation**  
   Specific sensors are manually disabled to simulate failure.

3. **Data Preprocessing**  
   Sensor values are normalized and filtered. Zero-value data is isolated for prediction.

4. **Model Training**  
   The ANN is trained using historical valid sensor data.

5. **Real-Time Prediction**  
   The model predicts missing sensor values with >95% accuracy.

---

## 📊 Performance Evaluation

| Scenario                     | Faulty Sensor   | Prediction Accuracy |
|-----------------------------|------------------|----------------------|
| Scenario 2                  | CO₂              | 95.0%                |
| Scenario 3                  | Light Sensor     | 95.9%                |
| Scenario 4                  | Humidity Sensor  | 99.99%               |
| Scenario 5                  | Temperature      | 99.99%               |

### 📈 Metrics
- **Relative Root Mean Square Error (rRMSE)**:  
  `0.023` *(example output, varies per run)*

- **Model Coefficient of Efficiency (C.E.)**:  
  `0.9994` *(very high predictive performance)*

---

## 📂 Repository Structure

```bash
AdaptoSense/
├── AdaptoSense_Model.py              # Final ANN implementation
├── AdaptoSenseProject Documentation.pdf
├── PresentationAdaptoSense.pptx
├── FinalNN_BTC.ipynb                 # Python training notebook
├── sensor_csv_files/                 # (CSV files used for training)
└── README.md                         # Project description and documentation
