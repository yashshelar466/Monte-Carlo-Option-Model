"""The Heston stochastic volatility model.

Black-Scholes assumes volatility is constant, which is why it cannot produce
the volatility smile seen in real option prices. Heston (1993) lets the
variance v itself move randomly:

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

    v0     starting variance (sqrt(v0) is today's volatility)
    kappa  how fast variance is pulled back towards its long-run level
    theta  long-run variance
    xi     volatility of variance ("vol of vol"): fattens both tails
    rho    correlation between price and variance moves; rho < 0 means
           volatility rises when the market falls, which tilts the smile
           into the skew seen in equity markets

Prices come from a semi-analytic Fourier formula, checked against a Monte
Carlo simulation of the two equations above.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from .monte_carlo import _payoff, _summarize


@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def as_tuple(self):
        return (self.v0, self.kappa, self.theta, self.xi, self.rho)


# Gauss-Legendre nodes for the Fourier integral, mapped from [-1, 1] to [0, U].
_U_MAX = 1000.0
_x, _w = np.polynomial.legendre.leggauss(1000)
_U = 0.5 * _U_MAX * (_x + 1)
_W = 0.5 * _U_MAX * _w


def _char_func(u, T, p):
    """Characteristic function of ln(S_T / S0) - (r - q) T under Heston.

    Uses the "little trap" form (Albrecher et al., 2007), which avoids the
    branch-cut jumps of Heston's original formula at long maturities.
    """
    v0, kappa, theta, xi, rho = p.as_tuple()
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def heston_price(S0, K, T, r, params, option="call", q=0.0):
    """Semi-analytic Heston price via the Lewis (2001) Fourier formula.

    K may be a scalar or an array of strikes (one expiry). Returns a float or
    an array to match.
    """
    K = np.asarray(K, dtype=float)
    x = np.log(S0 / K) + (r - q) * T                       # log-moneyness vs forward
    phi = _char_func(_U - 0.5j, T, params)                 # shape (n_nodes,)
    integrand = np.real(np.exp(1j * np.multiply.outer(x, _U)) * phi) / (_U**2 + 0.25)
    integral = integrand @ _W
    call = S0 * np.exp(-q * T) - np.sqrt(S0 * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    if option == "call":
        price = call
    elif option == "put":
        price = call - S0 * np.exp(-q * T) + K * np.exp(-r * T)  # put-call parity
    else:
        raise ValueError("option must be 'call' or 'put'")
    return float(price) if price.ndim == 0 else price


def simulate_heston(S0, T, r, params, n_paths, n_steps, q=0.0, seed=None):
    """Simulate Heston price paths (log-Euler for S, full-truncation Euler for v).

    Full truncation uses max(v, 0) wherever the variance enters the dynamics,
    which keeps the simulation stable when a step overshoots below zero.
    Antithetic pairs are used for variance reduction. Returns S, shape
    (n_paths, n_steps + 1).
    """
    v0, kappa, theta, xi, rho = params.as_tuple()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = (n_paths + 1) // 2
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    log_s = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, v0)
    for t in range(n_steps):
        z = rng.standard_normal((2, half))
        z = np.concatenate([z, -z], axis=1)[:, :n_paths]
        z1 = z[0]
        z2 = rho * z[0] + np.sqrt(1 - rho**2) * z[1]
        v_pos = np.maximum(v, 0.0)
        log_s += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
        v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2
        S[:, t + 1] = np.exp(log_s)
    return S


def mc_heston(S0, K, T, r, params, option="call", q=0.0, n_paths=100_000,
              n_steps=100, seed=None):
    """Monte Carlo price of a European option under Heston."""
    ST = simulate_heston(S0, T, r, params, n_paths, n_steps, q, seed)[:, -1]
    discounted = np.exp(-r * T) * _payoff(ST, K, option)
    return _summarize(discounted, antithetic=True)


def _bs_vega(S0, K, T, r, sigma, q):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def calibrate_heston(S0, strikes, T, r, market_prices, market_ivs, options, q=0.0,
                     initial=None):
    """Fit Heston parameters to one expiry's option prices.

    Minimises the squared differences in implied volatility, approximated as
    (model price - market price) / Black-Scholes vega, which is accurate for
    small errors and avoids inverting Black-Scholes at every step.
    `options` holds "call" or "put" per strike. Returns (HestonParams, rmse),
    where rmse is the root-mean-square implied-vol error.

    With a single expiry, kappa and theta trade off against each other, so
    several parameter sets can fit equally well; the smile is what is pinned down.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)
    is_call = np.asarray(options) == "call"
    vega = np.maximum(_bs_vega(S0, strikes, T, r, np.asarray(market_ivs), q), 1e-8)

    def residuals(x):
        p = HestonParams(*x)
        calls = heston_price(S0, strikes, T, r, p, "call", q)
        model = np.where(is_call, calls, calls - S0 * np.exp(-q * T) + strikes * np.exp(-r * T))
        return (model - market_prices) / vega

    atm_var = float(np.interp(S0, strikes, market_ivs)) ** 2
    x0 = initial.as_tuple() if initial else (atm_var, 2.0, atm_var, 0.5, -0.7)
    bounds = ([1e-4, 0.1, 1e-4, 0.01, -0.99], [1.0, 10.0, 1.0, 3.0, 0.99])
    fit = least_squares(residuals, x0, bounds=bounds, x_scale="jac")
    rmse = float(np.sqrt(np.mean(fit.fun**2)))
    return HestonParams(*map(float, fit.x)), rmse
