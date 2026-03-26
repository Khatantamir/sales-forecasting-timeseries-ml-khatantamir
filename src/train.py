import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import joblib
import os

# =========================
# 1. Load data
# =========================
df = pd.read_csv("data/raw/Walmart.csv")

# =========================
# 2. Data preprocessing
# =========================
df['Date'] = pd.to_datetime(df['Date'])

# зөвхөн Store 1 ашиглая (simple болгох)
df = df[df['Store'] == 1]

df = df.sort_values("Date")

# =========================
# 3. Feature Engineering (lag features)
# =========================
df['lag_1'] = df['Weekly_Sales'].shift(1)
df['lag_2'] = df['Weekly_Sales'].shift(2)
df['lag_3'] = df['Weekly_Sales'].shift(3)

# missing row устгах
df = df.dropna()

# =========================
# 4. Define X, y
# =========================
X = df[['lag_1', 'lag_2', 'lag_3']]
y = df['Weekly_Sales']

# =========================
# 5. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# 6. Train model
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 7. Predict
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. Evaluation
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# =========================
# 9. SAVE GRAPH (🔥 ЧУХАЛ)
# =========================
plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual Sales", linewidth=2)
plt.plot(y_pred, label="Predicted Sales", linestyle="--")

plt.title("Actual vs Predicted Weekly Sales")
plt.xlabel("Time (Weeks)")
plt.ylabel("Sales")

plt.legend()
plt.grid()

# зураг хадгалах
plt.savefig("forecast_plot.png")
plt.close()

print("Graph saved as forecast_plot.png")

# =========================
# 10. Save model
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sales_forecast_model.pkl")

print("Model saved successfully")
