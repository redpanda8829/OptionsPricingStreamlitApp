# import streamlit as st
# import numpy as np
# import math

# # Cumulative distribution function for the standard normal distribution
# def norm_cdf(x):
#     return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# st.title('Black-Scholes Options Price Calculator for European Call and Put Options')

# st.write('This app calculates the price of European call and put options using the Black-Scholes formula.')
# st.write('The Black-Scholes formula is given by:')
# st.latex(r'''C(S, K, r, \sigma, T) = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)''')
# st.latex(r'''P(S, K, r, \sigma, T) = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)''')
# st.write('Where:')
# st.latex(r'''d_1 = \frac{ln(S/K) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}''')
# st.latex(r'''d_2 = d_1 - \sigma \sqrt{T}''')
# st.write('Here,')
# st.latex(r'''S = \text{Current stock price}''')
# st.latex(r'''K = \text{Option strike price}''')
# st.latex(r'''r = \text{Risk-free interest rate}''')
# st.latex(r'''\sigma = \text{Stock price volatility}''')
# st.latex(r'''T = \text{Time to expiration}''')
# st.write('The cumulative distribution function of the standard normal distribution is denoted by $N(x)$.')

# # Variables
# S = st.number_input("Current Stock Price")
# K = st.number_input("Strike Price", min_value=0.0001)
# r = st.number_input("Risk-free rate")
# T = st.number_input("Time to maturity")
# sigma = st.number_input("Volatility")

# # Calculate d1 and d2
# d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# d2 = d1 - sigma * np.sqrt(T)

# # Call Option
# C = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
# # Put Option
# P = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

# # Print the results
# st.write("Call Option Price: $", round(C, 2))
# st.write("Put Option Price: $", round(P, 2))
