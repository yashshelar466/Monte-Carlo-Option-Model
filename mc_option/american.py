"""American options: Longstaff-Schwartz Monte Carlo and a binomial-tree benchmark.

An American option can be exercised at any time, so at every date the holder
compares exercising now against the value of waiting (the "continuation
value"). Simulation runs forward in time, which makes the continuation value
hard to know. Longstaff-Schwartz (2001) estimates it by regression: walk
backwards through the dates and, across all in-the-money paths, regress the
discounted cash flow from waiting on simple functions of today's stock price.
Exercise wherever the immediate payoff beats the fitted continuation value.
"""

import numpy as np

from .monte_carlo import _payoff, _summarize, simulate_gbm


def lsm_american(S0, K, T, r, sigma, option="put", q=0.0, n_paths=100_000,
                 n_steps=50, degree=3, seed=None, antithetic=True):
    """Price an American option by Longstaff-Schwartz least-squares Monte Carlo.

    Exercise is allowed on the n_steps simulated dates (a Bermudan
    approximation that approaches the American price as n_steps grows).
    The regression basis is 1, x, ..., x^degree with x = S / K.
    """
    paths = simulate_gbm(S0, T, r, sigma, n_paths, n_steps, q, seed, antithetic)
    disc = np.exp(-r * T / n_steps)

    # Cash flow each path receives under the exercise policy, valued at the
    # current date of the backward walk. Start with the payoff at expiry.
    cash = _payoff(paths[:, -1], K, option)
    for t in range(n_steps - 1, 0, -1):
        cash *= disc
        exercise = _payoff(paths[:, t], K, option)
        itm = np.flatnonzero(exercise > 0)
        if len(itm) <= degree + 1:
            continue
        # Regress only on in-the-money paths: those are the ones where the
        # exercise decision matters, and it keeps the fit accurate there.
        X = np.vander(paths[itm, t] / K, degree + 1)
        coef, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        continuation = X @ coef
        exercise_now = itm[exercise[itm] > continuation]
        cash[exercise_now] = exercise[exercise_now]

    result = _summarize(cash * disc, antithetic)
    # Exercising at time 0 is also allowed.
    result.price = max(result.price, float(_payoff(np.asarray(S0), K, option)))
    return result


def binomial_american(S0, K, T, r, sigma, option="put", q=0.0, n_steps=2_000):
    """Price an American option on a Cox-Ross-Rubinstein binomial tree.

    Deterministic and accurate for large n_steps, so it serves as the
    reference value for checking the Monte Carlo estimate.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    S = S0 * u ** np.arange(n_steps, -n_steps - 1, -2)
    value = _payoff(S, K, option)
    for _ in range(n_steps):
        S = S[:-1] / u
        value = np.maximum(disc * (p * value[:-1] + (1 - p) * value[1:]),
                           _payoff(S, K, option))
    return float(value[0])
