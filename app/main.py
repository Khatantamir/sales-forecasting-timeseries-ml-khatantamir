from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("models/sales_forecast_model.pkl")


@app.get("/")
def home():
    return {"message": "Sales Forecast API is running"}


@app.post("/predict")
def predict_sales(lag_1: float, lag_2: float, lag_3: float):

    X = np.array([[lag_1, lag_2, lag_3]])

    prediction = model.predict(X)

    return {
        "predicted_sales": float(prediction[0])
    }
