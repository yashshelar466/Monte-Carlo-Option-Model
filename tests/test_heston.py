import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from mc_option import bs_price
from mc_option.heston import (HestonParams, _char_func, calibrate_heston, heston_price,
                              mc_heston)
from mc_option.market import add_heston, implied_vol

PARAMS = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_reduces_to_black_scholes_when_volatility_is_constant(K):
    flat = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)
    assert heston_price(100, K, 1, 0.05, flat) == pytest.approx(bs_price(100, K, 1, 0.05, 0.2), abs=1e-6)


@pytest.mark.parametrize("T", [1 / 12, 1.0, 5.0])
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_fourier_integral_matches_adaptive_quadrature(T, K):
    r = 0.025
    x = np.log(100 / K) + r * T
    def f(u):
        return np.real(np.exp(1j * u * x) * _char_func(u - 0.5j, T, PARAMS)) / (u * u + 0.25)
    reference = 100 - np.sqrt(100 * K) * np.exp(-0.5 * r * T) / np.pi * quad(f, 0, np.inf, limit=500)[0]
    assert heston_price(100, K, T, r, PARAMS) == pytest.approx(reference, abs=1e-6)


def test_vectorised_strikes_and_put_call_parity():
    strikes = np.array([90.0, 100.0, 110.0])
    calls = heston_price(100, strikes, 0.5, 0.03, PARAMS, "call", q=0.01)
    puts = heston_price(100, strikes, 0.5, 0.03, PARAMS, "put", q=0.01)
    assert calls.shape == (3,)
    assert calls[1] == pytest.approx(heston_price(100, 100.0, 0.5, 0.03, PARAMS, "call", q=0.01))
    assert np.allclose(calls - puts, 100 * np.exp(-0.01 * 0.5) - strikes * np.exp(-0.03 * 0.5))


@pytest.mark.parametrize("K,option", [(90, "put"), (100, "call"), (110, "call")])
def test_monte_carlo_matches_formula(K, option):
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
    mc = mc_heston(100, K, 0.5, 0.03, params, option, n_paths=100_000, n_steps=50, seed=1)
    exact = heston_price(100, K, 0.5, 0.03, params, option)
    # Allow for the small time-discretisation bias of the Euler scheme.
    assert abs(mc.price - exact) < 4 * mc.std_error + 0.02


def test_negative_correlation_produces_equity_skew():
    S0, T, r = 100.0, 0.25, 0.03
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.8, rho=-0.8)
    iv = {K: implied_vol(heston_price(S0, K, T, r, params, "put"), S0, K, T, r, "put")
          for K in (80.0, 100.0, 120.0)}
    assert iv[80.0] > iv[100.0] > iv[120.0]


def spy_like_market():
    """A 30-day option chain priced with known Heston parameters."""
    S0, T, r = 762.63, 30 / 365, 0.0403
    true = HestonParams(v0=0.018, kappa=3.0, theta=0.035, xi=0.9, rho=-0.75)
    strikes = np.arange(615.0, 846.0, 5.0)
    options = np.where(strikes < S0, "put", "call")
    prices = np.array([heston_price(S0, k, T, r, true, o) for k, o in zip(strikes, options)])
    ivs = np.array([implied_vol(p, S0, k, T, r, o) for p, k, o in zip(prices, strikes, options)])
    return S0, T, r, true, strikes, options, prices, ivs


def test_calibration_recovers_known_parameters():
    S0, T, r, true, strikes, options, prices, ivs = spy_like_market()
    fit, rmse = calibrate_heston(S0, strikes, T, r, prices, ivs, options)
    assert rmse < 1e-6
    assert np.allclose(fit.as_tuple(), true.as_tuple(), rtol=1e-3)


def test_add_heston_fits_the_market_smile():
    S0, T, r, true, strikes, options, prices, ivs = spy_like_market()
    df = pd.DataFrame({"strike": strikes, "type": options, "market": prices, "implied_vol": ivs})
    out, params, rmse = add_heston(df, {"spot": S0, "T": T, "r": r})
    assert rmse < 1e-6
    assert np.allclose(out["heston"], prices, atol=1e-6)
    assert np.allclose(out["heston_iv"], ivs, atol=1e-6)
