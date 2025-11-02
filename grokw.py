import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Load the data
# Assuming the CSV is saved as 'AirQuality.csv'
# Replace with actual path if needed
df = pd.read_csv("D:\parth_ml\AirQuality.csv", delimiter=';', decimal=',')

# Step 2: Preprocessing
# Combine Date and Time into Datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
df = df.dropna(subset=['Datetime'])  # Drop rows with invalid datetime
df.set_index('Datetime', inplace=True)

# Replace -200 with NaN and impute with forward fill
df.replace(-200, np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)  # Backup fill for leading NaNs

# Select meteorological features and target pollutants
features = ['T', 'RH', 'AH']
targets = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']  # Example pollutants; adjust as needed

# Ensure selected columns exist and drop any remaining NaNs
df = df[features + targets].dropna()

# Step 3: Feature Engineering (Add lags and rolling means as per project)
for col in features + targets:
    df[f'{col}_lag1'] = df[col].shift(1)  # Lag 1
    df[f'{col}_rolling_mean_24'] = df[col].rolling(window=24).mean()  # 24-hour rolling mean

# Drop NaNs introduced by shifting/rolling
df.dropna(inplace=True)

# Update features to include engineered ones
engineered_features = [col for col in df.columns if col not in targets]

# Step 4: Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 5: Create sequences for LSTM (use past 24 hours to predict next)
def create_sequences(data, seq_length=24, feature_count=len(engineered_features), target_count=len(targets)):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :feature_count])  # Input sequences
        y.append(data[i+seq_length, feature_count:])    # Targets
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(scaled_data, seq_length)

# Step 6: Split into train/test (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 7: Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(len(targets)))  # Output for multiple targets
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Step 8: Evaluate the model
predictions = model.predict(X_test)

# Inverse scale predictions and y_test for accurate metrics
# Reconstruct full array for inverse transform
y_test_full = np.hstack((X_test[:, -1, :], y_test))  # Last timestep features + targets
pred_full = np.hstack((X_test[:, -1, :], predictions))

y_test_inv = scaler.inverse_transform(y_test_full)[:, -len(targets):]
pred_inv = scaler.inverse_transform(pred_full)[:, -len(targets):]

rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
mae = mean_absolute_error(y_test_inv, pred_inv)
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')

# Step 9: Visualization (Example: Plot actual vs predicted for first target)
plt.figure(figsize=(14, 5))
plt.plot(y_test_inv[:, 0], label='Actual CO(GT)')
plt.plot(pred_inv[:, 0], label='Predicted CO(GT)')
plt.title('Actual vs Predicted CO(GT)')
plt.xlabel('Time Steps')
plt.ylabel('CO(GT)')
plt.legend()
plt.show()

# Optional: Feature Importance Interpretation (using correlations from EDA)
corr_matrix = df.corr()
print("Correlations between meteo factors and pollutants:")
print(corr_matrix[targets].loc[features])

# To save the model
model.save('air_quality_lstm_model.h5')