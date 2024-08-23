#Black-Scholes Options Pricing Model
#Author: Azmain Yousuf
# Base case model for European Call and Put Options


import numpy as np
from scipy.stats import norm

#Variables
S = 42 #Current Stock Price
K = 40 #Strike Price
r = 0.044 #Risk-free rate
T = 0.5 #Time to maturity
sigma = 0.21 #Volatility

d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

#Call Option
C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#Put Option
P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

#Print the results
print("Call Option Price: $", round(C,2))
print("Put Option Price: $", round(P,2))