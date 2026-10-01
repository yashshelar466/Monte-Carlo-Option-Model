"""The Bates model: Heston stochastic volatility plus price jumps.

Heston moves continuously, so over a few weeks it cannot easily produce a big
drop, which is exactly what short-dated out-of-the-money puts insure against.
Bates (1996) adds sudden jumps to the stock price:

    dS/S = (r - q - lam k) dt + sqrt(v) dW1 + (J - 1) dN
    dv   = kappa (theta - v) dt + xi sqrt(v) dW2,       corr(dW1, dW2) = rho

    N is a Poisson process with lam jumps per year on average, and each jump
    multiplies the price by J, where ln J ~ Normal(mu_j, sigma_j^2).
    k = E[J - 1] = exp(mu_j + sigma_j^2 / 2) - 1 keeps the expected return at
    r - q, so prices stay arbitrage-free.

A negative mu_j (crashes) steepens the skew at short maturities without the
extreme volatility-of-volatility Heston needs on its own.
"""

from dataclasses import dataclass

import numpy as np

from .heston import HestonParams, _calibrate, _char_func, _fourier_price, simulate_heston
from .monte_carlo import _payoff, _summarize


@dataclass
class BatesParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lam: float       # average number of jumps per year
    mu_j: float      # mean log jump size (negative = crashes)
    sigma_j: float   # standard deviation of log jump size

    @property
    def heston(self):
        return HestonParams(self.v0, self.kappa, self.theta, self.xi, self.rho)

    def as_tuple(self):
        return (self.v0, self.kappa, self.theta, self.xi, self.rho,
                self.lam, self.mu_j, self.sigma_j)


def _jump_char_func(u, T, p):
    """Characteristic function of the compensated jump part of ln(S_T / S0)."""
    k = np.exp(p.mu_j + 0.5 * p.sigma_j**2) - 1
    jump = np.exp(1j * u * p.mu_j - 0.5 * p.sigma_j**2 * u**2) - 1
    return np.exp(p.lam * T * (jump - 1j * u * k))


def bates_char_func(u, T, p):
    """Characteristic function of ln(S_T / S0) - (r - q) T under Bates.

    Jumps are independent of the diffusion, so the two characteristic
    functions simply multiply.
    """
    return _char_func(u, T, p.heston) * _jump_char_func(u, T, p)


def bates_price(S0, K, T, r, params, option="call", q=0.0):
    """Semi-analytic Bates price via the Lewis (2001) Fourier formula."""
    return _fourier_price(S0, K, T, r, option, q, lambda u: bates_char_func(u, T, params))


def mc_bates(S0, K, T, r, params, option="call", q=0.0, n_paths=100_000, n_steps=100,
             seed=None):
    """Monte Carlo price of a European option under Bates."""
    ST = simulate_heston(S0, T, r, params.heston, n_paths, n_steps, q, seed,
                         jumps=(params.lam, params.mu_j, params.sigma_j))[:, -1]
    discounted = np.exp(-r * T) * _payoff(ST, K, option)
    return _summarize(discounted, antithetic=False)


def calibrate_bates(S0, strikes, T, r, market_prices, market_ivs, options, q=0.0,
                    initial=None):
    """Fit Bates parameters to one expiry's option prices.

    Starts from a Heston-like guess with occasional moderate crashes unless
    `initial` is given. Returns (BatesParams, implied-vol rmse).
    """
    if initial is None:
        atm_var = float(np.interp(S0, np.asarray(strikes, dtype=float), market_ivs)) ** 2
        initial = BatesParams(atm_var, 2.0, atm_var, 0.5, -0.7, 0.5, -0.1, 0.1)
    bounds = ([1e-4, 0.1, 1e-4, 0.01, -0.99, 0.0, -0.5, 0.005],
              [1.0, 10.0, 1.0, 3.0, 0.99, 10.0, 0.3, 0.5])
    x, rmse = _calibrate(lambda x, k: bates_price(S0, k, T, r, BatesParams(*x), "call", q),
                         initial.as_tuple(), bounds, S0, strikes, T, r, market_prices,
                         market_ivs, options, q)
    return BatesParams(*x), rmse
