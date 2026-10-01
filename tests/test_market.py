import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from mc_option import bs_price
from mc_option.market import (compare_to_market, with_implied_forward, fetch_market_data, historical_vol,
                              implied_vol, market_price, year_fraction)

S0, T, r = 100.0, 0.25, 0.04


def smile(K):
    """Volatility smile: higher implied vol for low strikes, like equity markets."""
    x = np.log(K / S0)
    return 0.2 - 0.1 * x + 0.5 * x**2


def synthetic_chain(option):
    strikes = np.arange(70.0, 131.0, 5.0)
    prices = np.array([bs_price(S0, k, T, r, smile(k), option) for k in strikes])
    return pd.DataFrame({"strike": strikes, "bid": prices - 0.05, "ask": prices + 0.05,
                         "lastPrice": prices})


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.1, 0.25, 0.8])
def test_implied_vol_round_trip(option, sigma):
    price = bs_price(S0, 110, T, r, sigma, option)
    assert implied_vol(price, S0, 110, T, r, option) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_impossible_prices():
    assert np.isnan(implied_vol(0.0, S0, 90, T, r, "call"))   # below intrinsic value
    assert np.isnan(implied_vol(150.0, S0, 90, T, r, "call"))  # above the stock price
    assert np.isnan(implied_vol(99.9, S0, 90, T, r, "call"))   # would need sigma > 500%


def test_historical_vol_recovers_sigma():
    rng = np.random.default_rng(0)
    returns = 0.3 / np.sqrt(252) * rng.standard_normal(20_000)
    closes = 100 * np.exp(np.cumsum(returns))
    assert historical_vol(closes) == pytest.approx(0.3, rel=0.02)


def test_market_price_falls_back_to_last_trade():
    chain = pd.DataFrame({"strike": [90, 100, 110], "bid": [1.0, 0.0, 0.0],
                          "ask": [1.2, 0.0, 0.0], "lastPrice": [1.1, 2.5, 0.0]})
    priced = market_price(chain)
    assert list(priced["price"]) == [1.1, 2.5]
    assert list(priced["price_source"]) == ["mid", "last"]


def test_year_fraction():
    now = datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)
    assert year_fraction("2027-01-01", now) == pytest.approx(1.0)
    assert year_fraction("2025-12-01", now) == 0.0


def test_compare_to_market_recovers_the_smile():
    data = {"spot": S0, "T": T, "r": r, "hist_vol": 0.2,
            "calls": synthetic_chain("call"), "puts": synthetic_chain("put")}
    df = compare_to_market(data, n_paths=20_000)
    assert list(df["strike"]) == sorted(df["strike"])
    assert set(df[df["strike"] < S0]["type"]) == {"put"}
    assert set(df[df["strike"] >= S0]["type"]) == {"call"}
    assert np.allclose(df["implied_vol"], smile(df["strike"]), atol=0.01)
    # At the money the smile equals the 20% historical vol, so model ~ market there.
    atm = df[df["strike"] == S0].iloc[0]
    assert abs(atm["model"] - atm["market"]) < 0.1


class FakeTicker:
    """Stands in for yfinance.Ticker so the tests run without internet access."""

    def __init__(self, symbol):
        self.symbol = symbol
        today = pd.Timestamp.now().normalize()
        self.options = tuple((today + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
                             for d in (7, 28, 91))

    def history(self, period):
        if self.symbol == "^IRX":
            return pd.DataFrame({"Close": [4.5, 5.0]})
        rng = np.random.default_rng(0)
        return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(0.0126 * rng.standard_normal(253)))})

    def option_chain(self, expiry):
        return types.SimpleNamespace(calls=synthetic_chain("call"), puts=synthetic_chain("put"))


@pytest.fixture
def fake_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))


def test_compare_to_market_skips_penny_options():
    data = {"spot": S0, "T": T, "r": r, "hist_vol": 0.2,
            "calls": synthetic_chain("call"), "puts": synthetic_chain("put")}
    all_rows = compare_to_market(data, min_price=0, n_paths=2_000)
    filtered = compare_to_market(data, min_price=0.2, n_paths=2_000)
    assert filtered["market"].min() >= 0.2
    assert set(all_rows["strike"]) - set(filtered["strike"]) == {80.0, 120.0}


def flat_vol_market(true_spot, q, sigma=0.2):
    """Calls and puts on every strike, priced at a flat vol with dividend yield q."""
    strikes = np.arange(80.0, 121.0, 1.0)
    def chain(option):
        prices = np.array([bs_price(true_spot, k, T, r, sigma, option, q) for k in strikes])
        return pd.DataFrame({"strike": strikes, "bid": prices - 0.01, "ask": prices + 0.01,
                             "lastPrice": prices})
    return chain("call"), chain("put")


def test_implied_forward_corrects_a_stale_spot_and_dividends():
    calls, puts = flat_vol_market(true_spot=100.0, q=0.012)
    stale = {"spot": 99.0, "T": T, "r": r, "hist_vol": 0.2, "calls": calls, "puts": puts}

    # With the stale spot, puts and calls disagree: a jump in implied vol at the money.
    before = compare_to_market(stale, n_paths=2_000)
    put_iv = before[before["type"] == "put"]["implied_vol"].iloc[-1]
    call_iv = before[before["type"] == "call"]["implied_vol"].iloc[0]
    assert call_iv - put_iv > 0.02

    fixed = with_implied_forward(stale)
    assert fixed["forward"] == pytest.approx(100.0 * np.exp((r - 0.012) * T), rel=1e-6)
    assert fixed["quoted_spot"] == 99.0
    after = compare_to_market(fixed, n_paths=2_000)
    assert np.allclose(after["implied_vol"], 0.2, atol=1e-4)


def test_implied_forward_without_matching_strikes_leaves_data_alone():
    calls, puts = flat_vol_market(true_spot=100.0, q=0.0)
    data = {"spot": 100.0, "T": T, "r": r, "calls": calls[calls["strike"] > 100],
            "puts": puts[puts["strike"] < 100]}
    out = with_implied_forward(data)
    assert out["forward"] is None and out["spot"] == 100.0


def test_fetch_market_data(fake_yfinance):
    data = fetch_market_data("FAKE", target_days=30)
    assert data["expiry"] == FakeTicker("FAKE").options[1]
    assert data["spot"] == FakeTicker("FAKE").history("1y")["Close"].iloc[-1]
    assert data["r"] == pytest.approx(0.05)
    assert 0.07 < data["T"] < 0.08
    assert 0.1 < data["hist_vol"] < 0.3


def test_market_demo_end_to_end(fake_yfinance, monkeypatch, tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg")
    import market_demo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["market_demo.py", "FAKE"])
    market_demo.main()
    assert (tmp_path / "market_smile.png").stat().st_size > 10_000
    out = capsys.readouterr().out
    assert "Saved market_smile.png" in out and "Implied forward" in out


def test_market_demo_with_heston(fake_yfinance, monkeypatch, tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg")
    import market_demo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["market_demo.py", "FAKE", "--heston"])
    market_demo.main()
    out = capsys.readouterr().out
    assert "Heston fit:" in out and "heston_iv" in out
    assert (tmp_path / "market_smile.png").exists()
