import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set style
sns.set(style="whitegrid")
np.random.seed(42)

# Simulate Smart Grid Energy Demand Signal
time = np.linspace(0, 24, 100)  # 24 hours
base_demand = 50 + 10 * np.sin(time * np.pi / 12)
noise = np.random.normal(0, 5, size=time.shape)
signal = base_demand + noise

# Risk management: simulate risk levels based on thresholds
risk_level = np.where(signal > 65, 1, 0)  # 1 = High Risk, 0 = Low Risk

# Create DataFrame
df = pd.DataFrame({
    'Time': time,
    'Demand': signal,
    'Risk': risk_level
})

# Adaptation & Mitigation: regression to forecast demand
X = df[['Time']]
y = df['Demand']
model = LinearRegression()
model.fit(X, y)
df['Predicted_Demand'] = model.predict(X)

# Classification model for risk
X_class = df[['Demand']]
y_class = df['Risk']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_class)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_class = np.where(y_pred > 0.5, 1, 0)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Smart Grid Framework: Risk Management, Adaptation & Mitigation", fontsize=16, color='darkblue')

# 1. Signal plot
axs[0, 0].plot(df['Time'], df['Demand'], label='Actual Demand', color='orange')
axs[0, 0].set_title("Energy Demand Signal")
axs[0, 0].set_xlabel("Time (Hours)")
axs[0, 0].set_ylabel("Demand (kWh)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Regression trend
axs[0, 1].plot(df['Time'], df['Demand'], label='Actual Demand', color='gray')
axs[0, 1].plot(df['Time'], df['Predicted_Demand'], label='Trend Line (Regression)', color='blue', linestyle='--')
axs[0, 1].set_title("Adaptation: Regression Forecast")
axs[0, 1].legend()

# 3. Risk classification
sns.scatterplot(x='Time', y='Demand', hue='Risk', palette={0: 'green', 1: 'red'}, data=df, ax=axs[1, 0])
axs[1, 0].set_title("Risk Management: High vs Low Risk")

# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk'])
disp.plot(ax=axs[1, 1], cmap='Blues', colorbar=False)
axs[1, 1].set_title("Mitigation Strategy: Risk Classification")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
