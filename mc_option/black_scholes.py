"""Closed-form Black-Scholes price, used to check the Monte Carlo results."""

import numpy as np
from scipy.stats import norm


def bs_price(S0, K, T, r, sigma, option="call", q=0.0):
    """Black-Scholes price of a European option with continuous dividend yield q."""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")
