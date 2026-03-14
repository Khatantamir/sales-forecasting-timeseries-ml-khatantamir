# Sales Forecasting Time Series ML System

## Project Overview

This project builds a machine learning system to forecast weekly retail sales using historical time-series data.

The goal is to predict future sales based on previous sales patterns.
The project demonstrates a complete data science workflow including:

* Exploratory Data Analysis (EDA)
* Time series feature engineering
* Machine learning model training
* Model evaluation
* Model deployment with FastAPI

This project is part of a Data Science portfolio.

---

## Dataset

Dataset: Walmart Store Sales Forecasting

The dataset contains weekly sales data for multiple Walmart stores.

Features include:

* Store
* Date
* Weekly_Sales
* Holiday_Flag
* Temperature
* Fuel_Price
* CPI
* Unemployment

Target variable:

Weekly_Sales

Dataset file:

data/raw/Walmart.csv

---

## Exploratory Data Analysis

EDA was performed in the notebook:

notebooks/eda.ipynb

Key steps:

* Load dataset
* Convert Date column to datetime
* Visualize weekly sales trends
* Explore time series patterns

Example time series plot:

Weekly sales were plotted over time to observe patterns and fluctuations.

---

## Feature Engineering

Time series forecasting requires lag features.

Lag features were created from historical sales:

lag_1 = previous week sales
lag_2 = sales two weeks ago
lag_3 = sales three weeks ago

These features allow the model to learn temporal dependencies.

Example:

Sales(t) → predicted using

Sales(t-1), Sales(t-2), Sales(t-3)

---

## Model Training

Model used:

RandomForestRegressor

Training script:

src/train.py

Training steps:

1. Load dataset
2. Filter Store 1
3. Create lag features
4. Split into training and test sets
5. Train Random Forest model
6. Save trained model

Saved model:

models/sales_forecast_model.pkl

---

## Model Evaluation

The model was evaluated using:

MAE — Mean Absolute Error
RMSE — Root Mean Squared Error

These metrics measure prediction accuracy for the test dataset.

---

## Prediction Script

Prediction script:

src/predict.py

Example input:

lag_1 = 1500000
lag_2 = 1480000
lag_3 = 1510000

Example output:

Predicted Sales: 1,520,000

---

## API Deployment

The model is deployed using FastAPI.

API file:

app/main.py

Start the API server:

uvicorn app.main:app --reload

API documentation will be available at:

http://127.0.0.1:8000/docs

Example API request:

POST /predict

Input:

{
"lag_1": 1500000,
"lag_2": 1480000,
"lag_3": 1510000
}

Output:

{
"predicted_sales": 1520000
}

---

## Project Structure

sales-forecasting-timeseries-ml-khatantamir

data
 raw
  Walmart.csv

models
 sales_forecast_model.pkl

notebooks
 eda.ipynb

src
 train.py
 predict.py

app
 main.py

requirements.txt
README.md

---

## Technologies Used

Python
Pandas
Scikit-learn
Matplotlib
Joblib
FastAPI
Uvicorn

---

## Author

Khatantamir Otgonbyamba

Data Science / Machine Learning Portfolio Project
