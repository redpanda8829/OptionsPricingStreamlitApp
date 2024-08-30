import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from matplotlib.colors import LinearSegmentedColormap

# Load the trained model
model = load_model('options_pricing_model.h5')

# Streamlit inputs for the parameters
st.title("Option Price Prediction Heatmap")
st.write("Use the sliders below to adjust the input parameters for the heatmap.")

S_min = st.slider("Minimum Stock Price", 50, 150, 50)
S_max = st.slider("Maximum Stock Price", 50, 150, 150)
sigma_min = st.slider("Minimum Volatility (sigma)", 0.1, 0.5, 0.1)
sigma_max = st.slider("Maximum Volatility (sigma)", 0.1, 0.5, 0.5)
K = st.number_input("Strike Price (K)", value=100)
r = st.number_input("Risk-free rate (r)", value=0.03)
T = st.number_input("Time to maturity (T)", value=1.0)

# Define ranges for stock price (S) and volatility (sigma) for a 10x10 grid
S_range = np.linspace(S_min, S_max, 10)
sigma_range = np.linspace(sigma_min, sigma_max, 10)

# Generate a meshgrid for S and sigma
S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)

# Flatten the grids to create input data
S_flat = S_grid.flatten()
sigma_flat = sigma_grid.flatten()

# Prepare the input data for prediction
input_data = np.column_stack((S_flat, np.full_like(S_flat, K), np.full_like(S_flat, r), sigma_flat, np.full_like(S_flat, T)))

# Predict all prices at once
predicted_prices = model.predict(input_data)

# Reshape the predictions back to the 10x10 grid shape
price_grid = predicted_prices.reshape(10, 10)
# Clip negative values to zero
price_grid = np.clip(price_grid, a_min=0, a_max=None)

# Custom color map from green to red
cmap = LinearSegmentedColormap.from_list("risk_cmap", ["green", "yellow", "red"])

# Plot the heatmap with annotations
plt.figure(figsize=(8, 8))
sns.heatmap(price_grid, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, linecolor='black',
            xticklabels=np.round(S_range, 2), yticklabels=np.round(sigma_range, 2), cbar_kws={'label': 'Option Price'})

plt.xlabel('Stock Price (S)')
plt.ylabel('Volatility (sigma)')
plt.title('Option Price Prediction Heatmap')

# Display the heatmap in Streamlit
st.pyplot(plt)
