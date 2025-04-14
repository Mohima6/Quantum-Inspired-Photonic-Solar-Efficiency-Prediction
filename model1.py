#Gradient Boosting Model for Solar Cell Efficiency Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# ---------------- RL ENV SETUP ---------------- #
np.random.seed(42)

# Define discrete action space for 4 features
bandgap = np.linspace(0.5, 2.0, 5)
absorp_coeff = np.linspace(1.0, 5.0, 5)
mobility = np.linspace(0.1, 0.5, 5)
temp = np.linspace(20, 45, 5)

actions = [(i, j, k, l) for i in bandgap for j in absorp_coeff for k in mobility for l in temp]

# Define reward function (simulate realistic formula)
def get_efficiency(b, a, m, t):
    noise = np.random.normal(0, 0.02)
    return 0.2*b + 0.3*a - 0.1*m + 0.05*t + noise

# Q-table
Q = np.zeros(len(actions))

# RL Parameters
alpha = 0.1
gamma = 0.95
episodes = 500

# RL: Q-Learning loop
for ep in range(episodes):
    idx = np.random.choice(len(actions))
    b, a, m, t = actions[idx]
    reward = get_efficiency(b, a, m, t)

    next_idx = np.random.choice(len(actions))
    Q[idx] = Q[idx] + alpha * (reward + gamma * np.max(Q[next_idx]) - Q[idx])

# Select best feature combo
best_idx = np.argmax(Q)
best_action = actions[best_idx]
print(f"🧠 Best Material Combo Found by RL: Bandgap={best_action[0]}, Absorption={best_action[1]}, Mobility={best_action[2]}, Temp={best_action[3]}")

# ---------------- ML MODELING ---------------- #
# Generate synthetic dataset
data = pd.DataFrame({
    'BandGap': np.random.choice(bandgap, 100),
    'AbsorptionCoefficient': np.random.choice(absorp_coeff, 100),
    'ChargeMobility': np.random.choice(mobility, 100),
    'Temperature': np.random.choice(temp, 100)
})

data['Efficiency'] = data.apply(lambda row: get_efficiency(row['BandGap'], row['AbsorptionCoefficient'], row['ChargeMobility'], row['Temperature']), axis=1)

# Train-test split
X = data.drop('Efficiency', axis=1)
y = data['Efficiency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# CatBoost model
model = CatBoostRegressor(verbose=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Save model
joblib.dump(model, "solar_efficiency_rl_model.pkl")

# Real-time prediction
sample_input = pd.DataFrame([{
    'BandGap': best_action[0],
    'AbsorptionCoefficient': best_action[1],
    'ChargeMobility': best_action[2],
    'Temperature': best_action[3]
}])
pred_eff = model.predict(sample_input)[0]
print(f"⚡ Predicted Efficiency from RL-Optimized Features: {pred_eff:.2f}%")

# ---------------- METRICS + VISUALIZATION ---------------- #
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:\nR² Score: {r2:.3f} | MSE: {mse:.4f}")

# Plot 1: Actual vs Predicted
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Predicted vs Actual Efficiency')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)

# Plot 2: Distribution
plt.subplot(2, 2, 2)
sns.kdeplot(y_pred, fill=True, color='green', label='Predicted')
sns.kdeplot(y_test, fill=True, color='red', label='Actual')
plt.title('Distribution Comparison')
plt.legend()

# Plot 3: Error
plt.subplot(2, 2, 3)
error = y_test - y_pred
plt.hist(error, bins=20, color='purple')
plt.title('Prediction Error Histogram')
plt.xlabel('Error')

# Plot 4: SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test[:50])
plt.subplot(2, 2, 4)
shap.plots.waterfall(shap_values[0], max_display=4, show=False)
plt.title("SHAP Waterfall (1st Test Sample)")

plt.tight_layout()
plt.show()

# ---------------- BD PANEL COMPARISON ---------------- #
bd_types = ['Mono', 'Poly', 'Thin Film']
bd_eff = [18.5, 16.0, 11.5]

plt.figure(figsize=(8, 5))
plt.plot(bd_types + ['RL-ML Model'], bd_eff + [pred_eff], marker='o', linewidth=2, color='darkblue')
plt.title("📈 Efficiency Comparison: BD Panels vs RL-ML Framework")
plt.ylabel("Efficiency (%)")
plt.grid(True)
plt.show()
