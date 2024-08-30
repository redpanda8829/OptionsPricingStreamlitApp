import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def generate_synthetic_data(num_samples=10000):
    S = np.random.uniform(50, 150, num_samples)
    K = np.random.uniform(50, 150, num_samples)
    r = np.random.uniform(0.01, 0.05, num_samples)
    T = np.random.uniform(0.1, 2, num_samples)
    sigma = np.random.uniform(0.1, 0.5, num_samples)

    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    X = np.stack((S, K, r, sigma, T), axis=1)
    y = C

    return X, y

X_train, y_train = generate_synthetic_data()
# Define the model architecture
model = Sequential([
    Dense(512, input_dim=5, activation='relu'),  # Input layer + hidden layer
    Dense(128, activation='relu'),               # Hidden layer
    Dense(1)                                    # Output layer (option price)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
X_test, y_test = generate_synthetic_data(num_samples=2000)  # Generate or load test data

# Evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

model.save('options_pricing_model.h5')

