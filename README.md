# Sales Forecasting Time Series ML System

A machine learning project that forecasts weekly retail sales using historical sales data.

This project is part of my Data Science / Machine Learning portfolio. It demonstrates a full workflow from data exploration to model training, evaluation, model saving, and API deployment.

## Project Goal

The goal of this project is to predict future weekly sales using past sales patterns.

This project includes:

- Data loading and cleaning
- Exploratory data analysis (EDA)
- Time series feature engineering
- Model training
- Model evaluation
- Model saving with joblib
- FastAPI prediction API

---

## Dataset

Dataset used: **Walmart Store Sales Forecasting**

Dataset file:

```
data/raw/Walmart.csv
```

Main columns:

- Store
- Date
- Weekly_Sales
- Holiday_Flag
- Temperature
- Fuel_Price
- CPI
- Unemployment

Target variable:

```
Weekly_Sales
```

---

## Problem Type

This is a **time series forecasting** problem.

The model uses previous weekly sales values to predict future weekly sales.

---

## Project Structure

```
sales-forecasting-timeseries-ml-khatantamir
│
├── app
│   └── main.py
│
├── data
│   └── raw
│       └── Walmart.csv
│
├── models
│   └── sales_forecast_model.pkl
│
├── notebooks
│   └── eda.ipynb
│
├── src
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis

EDA was performed in:

```
notebooks/eda.ipynb
```

Main steps:

- Load dataset using pandas
- Inspect first rows of the dataset
- Check dataset shape
- Inspect column names
- Convert `Date` column to datetime
- Plot weekly sales trends
- Analyze time series patterns

---

## Feature Engineering

Time series forecasting requires lag features.

Lag features created:

- lag_1 = previous week sales
- lag_2 = sales two weeks earlier
- lag_3 = sales three weeks earlier

These lag values allow the model to learn temporal patterns from past sales.

---

## Model Training

Training script:

```
src/train.py
```

Model used:

```
RandomForestRegressor
```

Training process:

1. Load dataset
2. Convert date column
3. Filter data for Store 1
4. Sort data by date
5. Create lag features
6. Remove missing rows
7. Split dataset into training and test sets
8. Train Random Forest model
9. Save trained model

---

## Model Evaluation

Model performance was evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

These metrics measure the difference between predicted sales and actual sales.

A visualization comparing actual and predicted sales was also generated.

---

## Saved Model

Trained model location:

```
models/sales_forecast_model.pkl
```

Library used for saving:

```
joblib
```

---

## Prediction Script

Prediction script:

```
src/predict.py
```

The script loads the saved model and predicts sales based on lag inputs.

Example input idea:

- previous week sales
- sales two weeks ago
- sales three weeks ago

The model returns the predicted weekly sales value.

---

## API Deployment

FastAPI application file:

```
app/main.py
```

Available endpoints:

GET `/`  
Health check endpoint

POST `/predict`  
Predict weekly sales

---

### Run the API

```
uvicorn app.main:app --reload
```

---

### Open API Documentation

```
http://127.0.0.1:8000/docs
```

---

## Example Prediction Input

Example lag values:

```
lag_1 = 1500000
lag_2 = 1480000
lag_3 = 1510000
```

Example output:

```
predicted_sales = 1520000
```

---

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- joblib
- FastAPI
- uvicorn

---

## Business Value

Sales forecasting helps businesses with:

- inventory planning
- staffing decisions
- budgeting
- demand forecasting
- strategic planning

Machine learning models can help companies make better data-driven decisions.

---

## Portfolio Value

This project demonstrates the ability to:

- work with time series datasets
- build lag features for forecasting
- train machine learning models
- evaluate prediction performance
- save models for reuse
- build a simple prediction API

---

## Author

Khatantamir Otgonbyamba

Data Science / Machine Learning Portfolio Project
