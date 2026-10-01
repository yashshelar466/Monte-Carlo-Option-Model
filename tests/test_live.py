"""Live checks against real Yahoo Finance data.

The other tests use a fake Yahoo module so they are fast and repeatable.
These make sure the real service still returns data in the shape the code
expects (columns, sensible values), which the fake cannot detect.

Skipped by default; run with:  pytest -m live
"""

import numpy as np
import pytest

from mc_option.market import compare_to_market, fetch_market_data, with_implied_forward

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def spy():
    return fetch_market_data("SPY", target_days=30)


def test_download_has_expected_shape(spy):
    assert spy["spot"] > 0
    assert 0 < spy["T"] < 0.5
    assert 0.02 < spy["hist_vol"] < 1.5
    assert 0 <= spy["r"] < 0.2
    for side in ("calls", "puts"):
        chain = spy[side]
        assert len(chain) > 20, f"too few {side} listed"
        missing = {"strike", "bid", "ask", "lastPrice"} - set(chain.columns)
        assert not missing, f"Yahoo {side} chain is missing columns {missing}"


def test_implied_forward_is_close_to_spot(spy):
    data = with_implied_forward(spy)
    assert data["forward"] is not None, "no strike has both a call and a put"
    assert abs(data["forward"] / spy["spot"] - 1) < 0.03


def test_market_comparison_produces_sensible_implied_vols(spy):
    df = compare_to_market(with_implied_forward(spy), n_paths=5_000)
    assert len(df) > 10
    iv = df["implied_vol"].dropna()
    assert len(iv) > 0.8 * len(df), "too many prices had no implied volatility"
    assert np.all((iv > 0.02) & (iv < 2.0))
