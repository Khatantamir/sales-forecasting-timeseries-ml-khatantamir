import joblib
import numpy as np

model = joblib.load("models/sales_forecast_model.pkl")

# example lag inputs
lag_1 = 1500000
lag_2 = 1480000
lag_3 = 1510000

X = np.array([[lag_1, lag_2, lag_3]])

prediction = model.predict(X)

print("Predicted Sales:", prediction[0])
