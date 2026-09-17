# SmartCanteen AI

SmartCanteen AI is an intelligent food demand prediction system designed to help canteen management predict the required quantity of food for the next day.

The system uses historical canteen sales data and Machine Learning to predict food demand and provide a recommended preparation quantity.

## 🚀 Live Demo

👉 https://smartcanteenai.onrender.com/

## 💻 GitHub Repository

👉 https://github.com/lalitbhade/SmartCanteenAI

## 🎯 Problem Statement

Canteens often face difficulty in estimating how much food should be prepared each day.

If too much food is prepared, it can lead to:

- Food wastage
- Increased cost
- Resource wastage

If too little food is prepared, it can lead to:

- Food shortages
- Unavailability of food items
- Student dissatisfaction

SmartCanteen AI helps solve this problem by predicting the expected demand for different food items.

## 💡 Solution

SmartCanteen AI uses historical sales data and Machine Learning to predict the expected demand for food items.

The system provides:

- Historical sales information
- Food-wise demand prediction
- Important prediction features
- Random Forest model information
- Predicted demand
- Safety buffer
- Recommended preparation quantity

## ✨ Features

### 1. Historical Data

The system displays historical canteen sales data that is used for Machine Learning.

### 2. Food-wise Prediction

The system predicts the expected demand for individual food items.

### 3. Feature Analysis

The system uses features such as:

- Day of week
- Students present
- Price
- Holiday
- Exam day
- Previous day sales
- 7-day average sales

### 4. Random Forest Regression

Random Forest Regression is used to predict food demand based on historical and contextual data.

### 5. Recommendation

The system calculates a recommended preparation quantity using the predicted demand and a configurable safety buffer.

### 6. Safety Buffer

An initial 8% safety buffer is used to handle small unexpected variations in demand.

Example:

Predicted Demand = 175

Safety Buffer = 8% of 175 = 14

Recommended Preparation = 175 + 14 = 189

The 8% value is configurable and can be changed for a real-world deployment.

### 7. Web Dashboard

The project provides a web-based dashboard built using Flask, HTML, CSS and JavaScript.

### 8. Mobile-Friendly Interface

The dashboard can be accessed from mobile phones, tablets and computers.

## 🔄 How It Works

The working flow of the system is:

Historical Canteen Sales
        ↓
Data Processing
        ↓
Feature Preparation
        ↓
Random Forest Regression
        ↓
Model Training
        ↓
Saved Machine Learning Model
        ↓
Flask Backend
        ↓
Web Dashboard
        ↓
Predicted Food Demand
        ↓
Recommended Preparation Quantity

## 🤖 Machine Learning Algorithm

The project uses:

**Random Forest Regression**

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to produce a prediction.

It is used in this project to estimate the demand for different food items based on historical sales and other relevant features.

## 📊 Example

Suppose the system predicts:

Predicted Demand = 175

Safety Buffer = 8%

Safety Buffer Quantity = 14

Recommended Preparation = 189

Therefore, the canteen can use the recommendation as an estimated preparation quantity for the next day.

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest Regression

### Data Processing

- Pandas
- NumPy

### Model Storage

- Joblib

### Backend

- Flask

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Render

## 📁 Project Structure

```text
SmartCanteenAI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── README.md
│
├── data/
│   └── sales.csv
│
├── models/
│   ├── idli.joblib
│   ├── misal_pav.joblib
│   ├── poha.joblib
│   ├── samosa.joblib
│   ├── tea.joblib
│   └── vada_pav.joblib
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css


