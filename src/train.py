import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# load data
df = pd.read_csv("data/raw/Walmart.csv")

# convert date
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# select store
store1 = df[df["Store"] == 1]

store1 = store1.sort_values("Date")

# lag features
store1["lag_1"] = store1["Weekly_Sales"].shift(1)
store1["lag_2"] = store1["Weekly_Sales"].shift(2)
store1["lag_3"] = store1["Weekly_Sales"].shift(3)

store1.dropna(inplace=True)

# train test split
train = store1[:-20]
test = store1[-20:]

features = ["lag_1","lag_2","lag_3"]

X_train = train[features]
y_train = train["Weekly_Sales"]

# model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# save model
joblib.dump(model, "models/sales_forecast_model.pkl")

print("Model trained and saved.")
