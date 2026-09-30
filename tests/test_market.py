import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from mc_option import bs_price
from mc_option.market import (compare_to_market, fetch_market_data, historical_vol,
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
    assert "Saved market_smile.png" in capsys.readouterr().out
