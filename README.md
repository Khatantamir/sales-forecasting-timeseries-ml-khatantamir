# Sales Forecasting Time Series ML System

A machine learning project that predicts weekly retail sales from past sales patterns.

This project is part of my Data Science / Machine Learning portfolio. It shows a full workflow from data analysis to model saving and API deployment.

## Project Goal

The goal of this project is to forecast weekly sales using historical Walmart sales data.

This project covers:

* Data loading and cleaning
* Exploratory data analysis
* Time series feature engineering
* Model training
* Model evaluation
* Model saving with joblib
* FastAPI prediction endpoint

## Dataset

Dataset used: Walmart Store Sales Forecasting

File used in this project:

`data/raw/Walmart.csv`

Main columns:

* `Store`
* `Date`
* `Weekly_Sales`
* `Holiday_Flag`
* `Temperature`
* `Fuel_Price`
* `CPI`
* `Unemployment`

Target variable:

* `Weekly_Sales`

## Problem Type

This is a **time series forecasting** problem.

The model uses past weekly sales values to predict future weekly sales.

## Project Structure

```
sales-forecasting-timeseries-ml-khatantamir/
│
├── app/
│   └── main.py
│
├── data/
│   └── raw/
│       └── Walmart.csv
│
├── models/
│   └── sales_forecast_model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

EDA was done in:

`notebooks/eda.ipynb`

Main steps:

* Loaded the dataset with pandas
* Checked the first rows
* Checked dataset shape
* Reviewed column names
* Converted `Date` to datetime format
* Plotted weekly sales over time
* Focused on one store for cleaner forecasting

## Feature Engineering

To make the data usable for forecasting, lag features were created.

Lag features used:

* `lag_1` = sales from 1 week earlier
* `lag_2` = sales from 2 weeks earlier
* `lag_3` = sales from 3 weeks earlier

These features help the model learn how past sales affect future sales.

## Model Training

The model was trained in:

`src/train.py`

Model used:

`RandomForestRegressor`

Training process:

1. Load the dataset
2. Convert the date column
3. Filter data for Store 1
4. Sort by date
5. Create lag features
6. Drop missing rows caused by shifting
7. Split data into train and test sets
8. Train the model
9. Save the trained model

## Model Evaluation

The model was evaluated with:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error

These metrics show how close predictions are to actual sales values.

A comparison plot of actual vs predicted sales was also created in the notebook.

## Saved Model

The trained model is saved here:

`models/sales_forecast_model.pkl`

Library used:

`joblib`

## Prediction Script

Prediction script:

`src/predict.py`

This script loads the saved model and predicts sales from example lag inputs.

Example inputs:

* previous week sales
* two weeks ago sales
* three weeks ago sales

The model then returns the predicted weekly sales.

## API Deployment

FastAPI app file:

`app/main.py`

The API includes:

* `GET /` → health check
* `POST /predict` → predict weekly sales

### Run the API

```bash
uvicorn app.main:app --reload
```

### Open API documentation

```
http://127.0.0.1:8000/docs
```

## Example Prediction Input

The prediction endpoint uses three lag values:

* `lag_1`
* `lag_2`
* `lag_3`

Example:

```
lag_1 = 1500000
lag_2 = 1480000
lag_3 = 1510000
```

Example output:

```
predicted_sales = 1520000
```

## Technologies Used

* Python
* pandas
* scikit-learn
* matplotlib
* joblib
* FastAPI
* uvicorn

## Why This Project Matters

Sales forecasting helps businesses with:

* inventory planning
* staffing decisions
* budgeting
* demand planning
* business strategy

This project demonstrates how machine learning can support business decision-making.

## Portfolio Value

This project demonstrates the ability to:

* work with time series data
* engineer forecasting features
* train machine learning models
* evaluate prediction performance
* save models for reuse
* deploy a simple prediction API

## Author

**Khatantamir Otgonbyamba**

Data Science / Machine Learning Portfolio Project
