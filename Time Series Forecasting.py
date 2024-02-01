#Time Series Forecasting_task_4
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

# Load data from CSV file
data = pd.read_csv('time_series_data.csv')

# Split data into training and testing sets
train_data = data[:1000]
test_data = data[1000:]

# Define the model
model = Sequential([
    LSTM(64, input_shape=(None, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_data, epochs=10)

# Test the model
predictions = model.predict(test_data)
