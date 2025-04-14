import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate synthetic data
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'plasmonic_np_size': np.random.uniform(20, 100, n_samples),  # nm
    'qd_bandgap': np.random.uniform(1.1, 2.0, n_samples),         # eV
    'phc_periodicity': np.random.uniform(200, 800, n_samples),    # nm
    'film_thickness': np.random.uniform(100, 1000, n_samples),    # nm
    'light_intensity': np.random.uniform(300, 1000, n_samples),   # W/m²
    'charge_efficiency': np.random.uniform(0.7, 1.0, n_samples),  # %
})

# Target (efficiency) with some physics-inspired formula + noise
data['efficiency'] = (
    0.1 * data['plasmonic_np_size'] +
    15 * data['qd_bandgap'] +
    0.03 * data['phc_periodicity'] +
    0.005 * data['film_thickness'] +
    0.01 * data['light_intensity'] +
    50 * data['charge_efficiency'] +
    np.random.normal(0, 5, n_samples)
) / 100  # Normalize to 0–1 range

# Features and target
X = data.drop(columns=['efficiency'])
y = data['efficiency']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------- Random Forest Model ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

# --------------------------- PyTorch Neural Network ------------------------
class SolarNN(nn.Module):
    def __init__(self, input_dim):
        super(SolarNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

input_dim = X_train.shape[1]
model = SolarNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Training loop
epochs = 200
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Predict
model.eval()
with torch.no_grad():
    y_nn_pred = model(X_test_tensor).numpy().flatten()

# --------------------------- Plotting All Results --------------------------
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: RF Actual vs Predicted
axs[0, 0].scatter(y_test, y_rf_pred, color='green', alpha=0.7, label='RF Predictions')
axs[0, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
axs[0, 0].set_title('Random Forest: Actual vs Predicted')
axs[0, 0].set_xlabel('Actual Efficiency')
axs[0, 0].set_ylabel('Predicted Efficiency')
axs[0, 0].legend()

# Plot 2: NN Actual vs Predicted
axs[0, 1].scatter(y_test, y_nn_pred, color='blue', alpha=0.7, label='NN Predictions')
axs[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
axs[0, 1].set_title('Neural Network: Actual vs Predicted')
axs[0, 1].set_xlabel('Actual Efficiency')
axs[0, 1].set_ylabel('Predicted Efficiency')
axs[0, 1].legend()

# Plot 3: RF Feature Importance
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=X.columns, ax=axs[1, 0], palette='viridis')
axs[1, 0].set_title('Feature Importance (Random Forest)')
axs[1, 0].set_xlabel('Importance')

# Plot 4: NN Training Loss Curve
axs[1, 1].plot(losses, color='purple')
axs[1, 1].set_title('Neural Network Training Loss')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('MSE Loss')

plt.tight_layout()
plt.show()
