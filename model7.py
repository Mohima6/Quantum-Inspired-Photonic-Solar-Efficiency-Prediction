#national grid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# === 1. Simulated National Grid Data ===
# Replace with real data from the national grid
np.random.seed(42)
time_stamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
power_demand = 1000 + 20 * np.arange(100) + np.random.normal(scale=50, size=100)
df = pd.DataFrame({'Time': time_stamps, 'Power_Demand': power_demand})

# === 2. AutoRegression Model ===
ar_model = AutoReg(df['Power_Demand'], lags=5).fit()
ar_predictions = ar_model.predict(start=5, end=len(df)-1)

# === 3. CatBoost Model ===
# Prepare supervised dataset
X = []
y = []

for i in range(5, len(df)):
    X.append(power_demand[i-5:i])
    y.append(power_demand[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_model = CatBoostRegressor(verbose=0)
cat_model.fit(X_train, y_train)
cat_predictions = cat_model.predict(X_test)

# === 4. Plotting ===
plt.figure(figsize=(12, 6))

# Plot actual demand
plt.plot(df['Time'][5:], df['Power_Demand'][5:], label='Actual Power Demand', color='black')

# Plot AR model predictions
plt.plot(df['Time'][5:], ar_predictions, label='AR Model Prediction', color='blue')

# Plot CatBoost model predictions
plt.scatter(df['Time'][X_test.shape[0]:X_test.shape[0]+len(cat_predictions)], cat_predictions, label='CatBoost Prediction', color='red')

plt.title("National Grid Power Demand Forecasting")
plt.xlabel("Time")
plt.ylabel("Power Demand (MW)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. Evaluation ===
print(f"AutoRegression RMSE: {np.sqrt(mean_squared_error(df['Power_Demand'][5:], ar_predictions)):.2f}")
print(f"CatBoost RMSE (on test set): {np.sqrt(mean_squared_error(y_test, cat_predictions)):.2f}")
