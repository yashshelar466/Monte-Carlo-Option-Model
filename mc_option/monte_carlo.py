"""Monte Carlo pricing under geometric Brownian motion.

Under the risk-neutral measure the stock follows
    dS = (r - q) S dt + sigma S dW
which has the exact solution
    S(t + dt) = S(t) * exp((r - q - sigma^2 / 2) dt + sigma sqrt(dt) Z),  Z ~ N(0, 1).

An option's price is the discounted expected payoff, exp(-rT) E[payoff(S)],
which we estimate by averaging the payoff over many simulated paths.
"""

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_price


@dataclass
class MCResult:
    price: float
    std_error: float

    @property
    def conf_int(self):
        """95% confidence interval for the price."""
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


def _normals(n_paths, n_steps, rng, antithetic):
    if antithetic:
        half = rng.standard_normal(((n_paths + 1) // 2, n_steps))
        return np.concatenate([half, -half])[:n_paths]
    return rng.standard_normal((n_paths, n_steps))


def simulate_gbm(S0, T, r, sigma, n_paths, n_steps, q=0.0, seed=None, antithetic=False):
    """Simulate GBM price paths. Returns an array of shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = _normals(n_paths, n_steps, rng, antithetic)
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def _payoff(S, K, option):
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def _summarize(discounted, antithetic):
    n = len(discounted)
    if antithetic:
        # Antithetic pairs are correlated, so estimate the error from pair averages.
        m = n // 2
        pairs = 0.5 * (discounted[:m] + discounted[m : 2 * m])
        return MCResult(float(discounted.mean()), float(pairs.std(ddof=1) / np.sqrt(m)))
    return MCResult(float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n)))


def mc_european(S0, K, T, r, sigma, option="call", q=0.0, n_paths=100_000,
                seed=None, antithetic=True, control_variate=True):
    """Price a European option.

    Only the terminal price matters, so we simulate it in a single step.
    With control_variate=True the discounted terminal stock price, whose
    expectation S0 * exp(-qT) is known exactly, is used to cancel noise.
    """
    ST = simulate_gbm(S0, T, r, sigma, n_paths, 1, q, seed, antithetic)[:, -1]
    discounted = np.exp(-r * T) * _payoff(ST, K, option)

    if control_variate:
        control = np.exp(-r * T) * ST
        cov = np.cov(discounted, control)
        beta = cov[0, 1] / cov[1, 1]
        discounted = discounted - beta * (control - S0 * np.exp(-q * T))

    return _summarize(discounted, antithetic)


def mc_asian(S0, K, T, r, sigma, option="call", q=0.0, n_paths=100_000, n_steps=252,
             seed=None, antithetic=True):
    """Price an arithmetic-average Asian option (no closed form exists).

    The average is taken over the monitoring dates, excluding S0.
    """
    paths = simulate_gbm(S0, T, r, sigma, n_paths, n_steps, q, seed, antithetic)
    avg = paths[:, 1:].mean(axis=1)
    discounted = np.exp(-r * T) * _payoff(avg, K, option)
    return _summarize(discounted, antithetic)


def mc_greeks(S0, K, T, r, sigma, option="call", q=0.0, n_paths=200_000, seed=0,
              dS=None, dsigma=0.01):
    """Delta, gamma and vega of a European option by central finite differences.

    Every bumped price reuses the same seed (common random numbers), so the
    simulation noise largely cancels in the differences.
    """
    dS = dS or 0.01 * S0

    def price(s, v):
        return mc_european(s, K, T, r, v, option, q, n_paths, seed,
                           antithetic=True, control_variate=False).price

    p0, up, down = price(S0, sigma), price(S0 + dS, sigma), price(S0 - dS, sigma)
    return {
        "delta": (up - down) / (2 * dS),
        "gamma": (up - 2 * p0 + down) / dS**2,
        "vega": (price(S0, sigma + dsigma) - price(S0, sigma - dsigma)) / (2 * dsigma),
    }


def compare_to_black_scholes(S0, K, T, r, sigma, option="call", q=0.0, **kwargs):
    """Return (Monte Carlo result, Black-Scholes price) for a sanity check."""
    return mc_european(S0, K, T, r, sigma, option, q, **kwargs), bs_price(S0, K, T, r, sigma, option, q)
