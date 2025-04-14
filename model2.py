#Neural Network for Solar Cell Efficiency Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import seaborn as sns

# 1. Generate synthetic solar cell dataset
np.random.seed(42)
data = pd.DataFrame({
    'BandGap': np.random.uniform(0.5, 2.0, 100),
    'AbsorptionCoefficient': np.random.uniform(1.0, 5.0, 100),
    'ChargeMobility': np.random.uniform(0.1, 0.5, 100),
    'Temperature': np.random.uniform(20, 45, 100)
})
data['Efficiency'] = (
    0.2 * data['BandGap'] +
    0.3 * data['AbsorptionCoefficient'] -
    0.1 * data['ChargeMobility'] +
    0.05 * data['Temperature'] +
    np.random.normal(0, 0.03, 100)
)

X = data.drop('Efficiency', axis=1)
y = data['Efficiency']

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model setup with cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
y_actual, y_pred_total = [], []

for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')

    history = model.fit(X_train, y_train, epochs=150, batch_size=8,
                        validation_data=(X_val, y_val), verbose=0)

    y_pred_fold = model.predict(X_val).flatten()
    y_pred_total.extend(y_pred_fold)
    y_actual.extend(y_val)

# 4. Evaluation
mse = mean_squared_error(y_actual, y_pred_total)
r2 = r2_score(y_actual, y_pred_total)

print(f"\n📈 Final MSE: {mse:.4f}")
print(f"🔍 R² Score: {r2:.4f}")

# 5. Plotting everything together
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🌞 Solar Cell Efficiency: Deep Learning Performance', fontsize=16)

# A. Predicted vs Actual
axs[0, 0].scatter(y_actual, y_pred_total, color='blue', alpha=0.6)
axs[0, 0].plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
axs[0, 0].set_title('Predicted vs Actual')
axs[0, 0].set_xlabel('Actual Efficiency')
axs[0, 0].set_ylabel('Predicted Efficiency')

# B. Residual Plot
residuals = np.array(y_actual) - np.array(y_pred_total)
sns.histplot(residuals, kde=True, ax=axs[0, 1], color='purple')
axs[0, 1].set_title('Residual Distribution')
axs[0, 1].set_xlabel('Residual')

# C. Training Loss (last fold)
axs[1, 0].plot(history.history['loss'], label='Train Loss')
axs[1, 0].plot(history.history['val_loss'], label='Val Loss')
axs[1, 0].set_title('Training vs Validation Loss')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

# D. Permutation Feature Importance (using surrogate model)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_scaled, y)
importances = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42)
features = X.columns
axs[1, 1].barh(features, importances.importances_mean, color='green')
axs[1, 1].set_title('Feature Importance (Permutation)')

plt.tight_layout()
plt.show()
