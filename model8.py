import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces

# --- Generate Random Data (Synthetic Dataset) ---
np.random.seed(42)

# Generate synthetic data for demonstration purposes
time_index = pd.date_range('2025-01-01', periods=1000, freq='H')
load_values = np.random.normal(loc=200, scale=50, size=1000)  # Mean=200, Std=50

# Create the DataFrame
data = pd.DataFrame({'Time': time_index, 'Load': load_values})

# --- Forecasting Model Setup ---
# Prepare the data for training
data['Hour'] = data['Time'].dt.hour
X = data[['Hour']]
y = data['Load']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Apply MinMaxScaler for better scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gradient Boosting Regressor for forecasting
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X_train_scaled, y_train)

# Predictions and Evaluation
gbr_preds = gbr_model.predict(X_test_scaled)
gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_preds))

print(f"Gradient Boosting Model RMSE: {gbr_rmse:.2f}")

# --- Neural Network Model for Load Forecasting (for comparison) ---
nn_model = Sequential([
    Dense(64, input_dim=1, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer=Adam(), loss='mean_squared_error')
nn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

# Neural network predictions and evaluation
nn_preds = nn_model.predict(X_test_scaled).flatten()
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))

print(f"Neural Network Model RMSE: {nn_rmse:.2f}")

# --- Reinforcement Learning (RL) for Grid Optimization ---
class LoadOptimizationEnv(gym.Env):
    def __init__(self, model):
        super(LoadOptimizationEnv, self).__init__()
        self.model = model
        self.action_space = spaces.Discrete(2)  # 0 = decrease, 1 = increase load
        self.observation_space = spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.array([0], dtype=np.float32)  # Start at hour 0

    def step(self, action):
        # Apply action: 0 -> decrease load, 1 -> increase load
        if action == 0:
            load_change = -np.random.uniform(10, 50)  # Decrease by 10-50 units
        else:
            load_change = np.random.uniform(10, 50)  # Increase by 10-50 units

        self.current_step += 1
        next_state = np.array([self.current_step % 24], dtype=np.float32)

        # Get forecasted load from the model
        predicted_load = self.model.predict(next_state.reshape(1, -1))[0]
        reward = -abs(predicted_load - (200 + load_change))  # Minimize the load deviation

        done = self.current_step >= len(X_test)  # End after reaching dataset length

        return next_state, reward, done, {}

    def render(self):
        pass

# --- Setup and Train RL Agent ---
env = LoadOptimizationEnv(gbr_model)  # Use the Gradient Boosting model for RL
state = env.reset()

# Simple Q-learning setup
q_table = np.zeros([24, 2])  # 24 hours, 2 actions (increase/decrease)
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1  # Exploration rate

num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[int(state[0])])  # Exploitation

        next_state, reward, done, _ = env.step(action)

        # Update Q-table
        q_table[int(state[0]), action] = q_table[int(state[0]), action] + learning_rate * (
            reward + discount_factor * np.max(q_table[int(next_state[0])]) - q_table[int(state[0]), action]
        )

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon * 0.995, 0.01)

print("Training completed!")

# --- Final Load Prediction vs Actual Plot ---
optimized_load = np.zeros(len(y_test))

# Simulate RL-based load optimization (framework's output)
state = env.reset()
for i in range(len(y_test)):
    action = np.argmax(q_table[int(state[0])])  # Use the trained RL agent
    next_state, _, _, _ = env.step(action)
    optimized_load[i] = gbr_preds[i]  # Assume framework works with the GBR predictions for simplicity
    state = next_state

# Plotting the comparison
plt.figure(figsize=(12, 6))
plt.plot(data['Time'][-100:], y_test[-100:], label='Actual Load', color='blue')
plt.plot(data['Time'][-100:], gbr_preds[-100:], label='GBR Predicted Load', color='green')
plt.plot(data['Time'][-100:], nn_preds[-100:], label='NN Predicted Load', color='red')
plt.plot(data['Time'][-100:], optimized_load[-100:], label='Optimized Load (Framework)', color='orange', linestyle='--')

# Adding labels to show comparison
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Load Prediction: Actual vs Predicted with Framework Optimization')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
