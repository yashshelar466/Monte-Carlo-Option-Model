"""Convergence study: how the Monte Carlo estimate settles as paths increase."""

import numpy as np

from .black_scholes import bs_price
from .monte_carlo import mc_european


def convergence_study(S0, K, T, r, sigma, option="call", q=0.0,
                      path_counts=None, seed=0):
    """Price a European option at increasing path counts, with and without
    variance reduction.

    Each path count uses its own seed, so the estimates are independent runs
    rather than one growing sample. Returns a dict of NumPy arrays plus the
    Black-Scholes reference price.
    """
    if path_counts is None:
        path_counts = np.unique(np.logspace(2, 6, 25).astype(int))
    path_counts = np.asarray(path_counts)

    out = {"n_paths": path_counts, "bs_price": bs_price(S0, K, T, r, sigma, option, q)}
    for name, kwargs in (("plain", dict(antithetic=False, control_variate=False)),
                         ("reduced", dict(antithetic=True, control_variate=True))):
        results = [mc_european(S0, K, T, r, sigma, option, q, n_paths=int(n),
                               seed=seed + i, **kwargs)
                   for i, n in enumerate(path_counts)]
        out[f"{name}_price"] = np.array([res.price for res in results])
        out[f"{name}_se"] = np.array([res.std_error for res in results])
    return out
