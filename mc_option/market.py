"""Compare the model with real option prices.

Two ideas connect the model to the market:

* Historical volatility: the standard deviation of the stock's past daily log
  returns, annualised. Plugging it into the model gives a "fair" price based
  on how much the stock has actually moved.
* Implied volatility: the sigma that makes Black-Scholes reproduce an option's
  market price. It is the market's own volatility forecast. If Black-Scholes
  were exactly right, every strike would imply the same sigma; in practice
  out-of-the-money puts imply more, which draws a "volatility smile".
"""

from datetime import datetime, time, timezone

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .black_scholes import bs_price
from .heston import calibrate_heston, heston_price
from .monte_carlo import mc_european


def implied_vol(price, S0, K, T, r, option="call", q=0.0):
    """Sigma at which the Black-Scholes price equals `price`; NaN if no such sigma."""
    fwd_S, fwd_K = S0 * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(fwd_S - fwd_K, 0.0) if option == "call" else max(fwd_K - fwd_S, 0.0)
    upper = fwd_S if option == "call" else fwd_K
    if not lower < price < upper:
        return float("nan")
    try:
        return brentq(lambda v: bs_price(S0, K, T, r, v, option, q) - price, 1e-4, 5.0)
    except ValueError:  # needs a volatility outside 0.01%-500%
        return float("nan")


def historical_vol(closes, periods_per_year=252):
    """Annualised volatility from a series of closing prices."""
    log_returns = np.diff(np.log(np.asarray(closes, dtype=float)))
    return float(log_returns.std(ddof=1) * np.sqrt(periods_per_year))


def market_price(chain):
    """Add a `price` column: bid/ask midpoint, or last trade when no live quote.

    Outside market hours quotes are often zero, so the last trade is the only
    price available; `price_source` records which one was used.
    """
    chain = chain.copy()
    quoted = (chain["bid"] > 0) & (chain["ask"] > 0)
    chain["price"] = np.where(quoted, (chain["bid"] + chain["ask"]) / 2, chain["lastPrice"])
    chain["price_source"] = np.where(quoted, "mid", "last")
    return chain[chain["price"] > 0]


def year_fraction(expiry, now=None):
    """Years from `now` until 4pm New York time (~20:00 UTC) on the expiry date."""
    now = now or datetime.now(timezone.utc)
    expiry_dt = datetime.combine(pd.Timestamp(expiry).date(), time(20, 0), timezone.utc)
    return max((expiry_dt - now).total_seconds(), 0.0) / (365 * 24 * 3600)


def fetch_market_data(ticker, expiry=None, target_days=30, history="1y"):
    """Download spot, an option chain, historical vol and a risk-free rate from Yahoo Finance.

    Picks the listed expiry closest to `target_days` away unless `expiry`
    ("YYYY-MM-DD") is given. Needs the optional `yfinance` package.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    closes = tk.history(period=history)["Close"].dropna()
    if closes.empty:
        raise ValueError(f"No price history for {ticker!r}")
    if not tk.options:
        raise ValueError(f"No listed options for {ticker!r}")
    if expiry is None:
        today = pd.Timestamp.now().normalize()
        expiry = min(tk.options, key=lambda d: abs((pd.Timestamp(d) - today).days - target_days))

    chain = tk.option_chain(expiry)
    return {
        "ticker": ticker,
        "spot": float(closes.iloc[-1]),
        "expiry": expiry,
        "T": year_fraction(expiry),
        "hist_vol": historical_vol(closes),
        "r": risk_free_rate(),
        "calls": chain.calls,
        "puts": chain.puts,
    }


def risk_free_rate(default=0.04):
    """13-week US Treasury bill yield (Yahoo symbol ^IRX), or `default` if unavailable."""
    try:
        import yfinance as yf

        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(irx.iloc[-1]) / 100 if not irx.empty else default
    except Exception:
        return default


def with_implied_forward(data, n_strikes=10):
    """Replace the quoted spot with the one the option prices themselves imply.

    Put-call parity, C - P = (F - K) e^(-rT), gives the forward price F from
    any strike that has both a call and a put. Using the median over the
    strikes nearest the money removes two problems at once: a spot price that
    lags the option quotes, and an unknown dividend yield (F already includes
    it). European prices depend on the spot only through F, so pricing with
    spot F e^(-rT) and q = 0 is then exact.

    Returns a copy of `data` with the adjusted `spot`, plus `quoted_spot`,
    `forward` and `implied_q` (the dividend yield the forward implies). If no
    strike has both a call and a put, `data` is returned unchanged with
    `forward` set to None.
    """
    S0, T, r = data["spot"], data["T"], data["r"]
    calls = market_price(data["calls"])[["strike", "price"]]
    puts = market_price(data["puts"])[["strike", "price"]]
    both = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
    if both.empty or T <= 0:
        return {**data, "forward": None}
    nearest = both.iloc[(both["strike"] - S0).abs().argsort()[:n_strikes]]
    forwards = nearest["strike"] + (nearest["price_call"] - nearest["price_put"]) * np.exp(r * T)
    F = float(forwards.median())
    return {**data, "spot": F * np.exp(-r * T), "quoted_spot": S0, "forward": F,
            "implied_q": r - np.log(F / S0) / T}


def compare_to_market(data, q=0.0, moneyness=(0.8, 1.2), min_price=0.05,
                      n_paths=50_000, seed=0):
    """Price each out-of-the-money option with the model and back out its implied vol.

    Uses puts below the spot and calls above it, the liquid side of each
    strike. Options priced below `min_price` are skipped: at a cent or two the
    price is mostly tick size, so the implied vol it gives is noise.
    Returns a DataFrame sorted by strike.
    """
    S0, T, r, sigma = data["spot"], data["T"], data["r"], data["hist_vol"]
    if T <= 0:
        raise ValueError(f"Expiry {data.get('expiry')} has passed; choose a later one")
    lo, hi = moneyness[0] * S0, moneyness[1] * S0
    sides = [(market_price(data["puts"]), "put", lambda k: (k >= lo) & (k < S0)),
             (market_price(data["calls"]), "call", lambda k: (k >= S0) & (k <= hi))]

    rows = []
    for chain, option, in_range in sides:
        chain = chain[chain["price"] >= min_price]
        for row in chain[in_range(chain["strike"])].itertuples():
            model = mc_european(S0, row.strike, T, r, sigma, option, q, n_paths, seed)
            rows.append({
                "strike": row.strike,
                "type": option,
                "market": row.price,
                "source": row.price_source,
                "model": model.price,
                "model_se": model.std_error,
                "implied_vol": implied_vol(row.price, S0, row.strike, T, r, option, q),
            })
    columns = ["strike", "type", "market", "source", "model", "model_se", "implied_vol"]
    return pd.DataFrame(rows, columns=columns).sort_values("strike", ignore_index=True)


def add_heston(df, data, q=0.0):
    """Calibrate Heston to the market implied vols in `df` and add its prices and vols.

    Returns (df with `heston` and `heston_iv` columns, HestonParams, IV rmse).
    """
    S0, T, r = data["spot"], data["T"], data["r"]
    fit_rows = df.dropna(subset=["implied_vol"])
    if len(fit_rows) < 5:
        raise ValueError("Need at least 5 options with implied vols to calibrate Heston")
    params, rmse = calibrate_heston(S0, fit_rows["strike"], T, r, fit_rows["market"],
                                    fit_rows["implied_vol"], fit_rows["type"], q)
    df = df.copy()
    df["heston"] = [heston_price(S0, k, T, r, params, o, q) for k, o in zip(df["strike"], df["type"])]
    df["heston_iv"] = [implied_vol(p, S0, k, T, r, o, q)
                       for p, k, o in zip(df["heston"], df["strike"], df["type"])]
    return df, params, rmse
