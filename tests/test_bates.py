from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from mc_option.bates import BatesParams, bates_char_func, bates_price, calibrate_bates, mc_bates
from mc_option.heston import HestonParams, calibrate_heston, heston_price
from mc_option.market import add_bates, implied_vol

HESTON = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
BATES = BatesParams(*HESTON.as_tuple(), lam=0.8, mu_j=-0.12, sigma_j=0.15)


def test_without_jumps_bates_is_heston():
    no_jumps = BatesParams(*HESTON.as_tuple(), lam=0.0, mu_j=-0.1, sigma_j=0.1)
    strikes = np.array([80.0, 100.0, 120.0])
    assert np.allclose(bates_price(100, strikes, 0.5, 0.03, no_jumps, "put"),
                       heston_price(100, strikes, 0.5, 0.03, HESTON, "put"), atol=1e-12)


@pytest.mark.parametrize("T", [1 / 12, 1.0])
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_fourier_integral_matches_adaptive_quadrature(T, K):
    r = 0.03
    x = np.log(100 / K) + r * T
    def f(u):
        return np.real(np.exp(1j * u * x) * bates_char_func(u - 0.5j, T, BATES)) / (u * u + 0.25)
    reference = 100 - np.sqrt(100 * K) * np.exp(-0.5 * r * T) / np.pi * quad(f, 0, np.inf, limit=500)[0]
    assert bates_price(100, K, T, r, BATES) == pytest.approx(reference, abs=1e-6)


@pytest.mark.parametrize("K,option", [(80, "put"), (100, "call"), (120, "call")])
def test_monte_carlo_matches_formula(K, option):
    mc = mc_bates(100, K, 0.5, 0.03, BATES, option, n_paths=100_000, n_steps=50, seed=2)
    exact = bates_price(100, K, 0.5, 0.03, BATES, option)
    assert abs(mc.price - exact) < 4 * mc.std_error + 0.02


def test_downward_jumps_make_crash_protection_expensive():
    put_bates = bates_price(100, 80, 0.25, 0.03, BATES, "put")
    put_heston = heston_price(100, 80, 0.25, 0.03, HESTON, "put")
    assert put_bates > 1.5 * put_heston


def test_calibration_fits_a_known_bates_smile():
    S0, T, r = 100.0, 30 / 365, 0.04
    true = BatesParams(0.02, 3.0, 0.04, 0.8, -0.6, lam=0.3, mu_j=-0.15, sigma_j=0.12)
    strikes = np.arange(80.0, 116.0, 1.0)
    options = np.where(strikes < S0, "put", "call")
    prices = bates_price(S0, strikes, T, r, true, "call")
    prices = np.where(options == "call", prices, prices - S0 + strikes * np.exp(-r * T))
    ivs = np.array([implied_vol(p, S0, k, T, r, o) for p, k, o in zip(prices, strikes, options)])
    fit, rmse = calibrate_bates(S0, strikes, T, r, prices, ivs, options)
    assert rmse < 1e-3  # the smile is reproduced, even if parameters trade off
    assert fit.rho < 0 and fit.mu_j < 0


def load_spy():
    """Live SPY prices recorded on 1 Oct 2026 (see tests/data)."""
    df = pd.read_csv(Path(__file__).parent / "data" / "spy_2026-10-01.csv", comment="#")
    T, r = 0.080, 0.0399
    # Spot implied by put-call parity at the money (763 put vs 764/765 calls, interpolated).
    c763 = 13.69 + (13.69 - 13.02)
    S0 = (763 + (c763 - 9.98) * np.exp(r * T)) * np.exp(-r * T)
    df["implied_vol"] = [implied_vol(p, S0, k, T, r, o)
                         for p, k, o in zip(df["price"], df["strike"], df["type"])]
    return df.rename(columns={"price": "market"}), {"spot": S0, "T": T, "r": r}


def test_bates_fits_real_spy_skew_far_better_than_heston():
    df, data = load_spy()
    S0, T, r = data["spot"], data["T"], data["r"]
    _, heston_rmse = calibrate_heston(S0, df["strike"], T, r, df["market"], df["implied_vol"], df["type"])
    out, params, bates_rmse = add_bates(df, data)
    assert heston_rmse > 0.005           # Heston misses by ~0.7 vol points
    assert bates_rmse < 0.002            # Bates fits to ~0.05
    assert params.lam > 0 and params.mu_j < 0   # the market prices in crashes
    far_put = out[out["strike"] == 615].iloc[0]
    assert abs(far_put["bates_iv"] - far_put["implied_vol"]) < 0.01
