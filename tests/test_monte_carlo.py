import numpy as np
import pytest
from scipy.stats import norm

from mc_option import bs_price, mc_asian, mc_european, mc_greeks, simulate_gbm

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("antithetic,cv", [(False, False), (True, False), (True, True)])
def test_european_matches_black_scholes(option, antithetic, cv):
    res = mc_european(S0, K, T, r, sigma, option, n_paths=200_000, seed=1,
                      antithetic=antithetic, control_variate=cv)
    assert abs(res.price - bs_price(S0, K, T, r, sigma, option)) < 4 * res.std_error


def test_variance_reduction_lowers_error():
    plain = mc_european(S0, K, T, r, sigma, seed=1, antithetic=False, control_variate=False)
    reduced = mc_european(S0, K, T, r, sigma, seed=1)
    assert reduced.std_error < 0.8 * plain.std_error


def test_gbm_terminal_mean():
    paths = simulate_gbm(S0, T, r, sigma, 200_000, 10, seed=2)
    assert paths.shape == (200_000, 11)
    assert np.allclose(paths[:, 0], S0)
    assert abs(paths[:, -1].mean() - S0 * np.exp(r * T)) < 0.5


def test_asian_cheaper_than_european():
    assert mc_asian(S0, K, T, r, sigma, n_steps=50, seed=3).price < bs_price(S0, K, T, r, sigma)


def test_greeks_match_black_scholes():
    g = mc_greeks(S0, K, T, r, sigma)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    assert g["delta"] == pytest.approx(norm.cdf(d1), abs=0.01)
    assert g["vega"] == pytest.approx(S0 * norm.pdf(d1) * np.sqrt(T), rel=0.03)
    assert g["gamma"] == pytest.approx(norm.pdf(d1) / (S0 * sigma * np.sqrt(T)), rel=0.15)
