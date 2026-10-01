"""Compare model prices with live option prices and plot the volatility smile.

Needs internet access and yfinance (pip install -r requirements.txt).

    python market_demo.py                 # SPY, expiry about 30 days out
    python market_demo.py AAPL --days 60
    python market_demo.py ^SPX --q 0.013  # S&P 500 index options (European)
    python market_demo.py --min-price 0   # keep even penny-priced options
    python market_demo.py --heston        # also fit the Heston model to the smile

Saves market_smile.png.
"""

import argparse

import matplotlib.pyplot as plt

from mc_option.market import add_heston, compare_to_market, fetch_market_data

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MARKET, MODEL, HESTON, REFERENCE = "#eb6834", "#2a78d6", "#1baf7a", "#52514e"


def plot(df, data, path="market_smile.png", heston=None):
    S0, sigma = data["spot"], data["hist_vol"]
    plt.rcParams.update({
        "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
        "axes.edgecolor": GRID, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(S0, color=REFERENCE, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Strike")

    ax1.plot(df["strike"], df["model"], color=MODEL, linewidth=2,
             label=f"Model (historical vol {sigma:.1%})")
    if heston is not None:
        ax1.plot(df["strike"], df["heston"], color=HESTON, linewidth=2, linestyle="--",
                 label="Heston (fitted)")
    ax1.scatter(df["strike"], df["market"], color=MARKET, s=28, zorder=3,
                edgecolors=SURFACE, linewidths=1, label="Market price")
    ax1.set_ylabel("Option price")
    ax1.set_title("Out-of-the-money puts (left of spot) and calls (right)",
                  loc="left", fontsize=11, color=TEXT)
    ax1.legend(frameon=False, loc="upper left")

    iv = df.dropna(subset=["implied_vol"])
    ax2.scatter(iv["strike"], iv["implied_vol"] * 100, color=MARKET, s=28, zorder=3,
                edgecolors=SURFACE, linewidths=1, label="Market implied volatility")
    if heston is not None:
        hiv = df.dropna(subset=["heston_iv"])
        params, rmse = heston
        ax2.plot(hiv["strike"], hiv["heston_iv"] * 100, color=HESTON, linewidth=2,
                 linestyle="--", label=f"Heston (fit error {rmse:.2%} vol)")
    ax2.axhline(sigma * 100, color=REFERENCE, linestyle="--", linewidth=1.2,
                label=f"Historical volatility {sigma:.1%}")
    ax2.set_ylabel("Volatility (%)")
    ax2.set_title("Volatility smile: the market's sigma varies by strike",
                  loc="left", fontsize=11, color=TEXT)
    ax2.legend(frameon=False, loc="upper right")
    ax2.annotate(f"spot {S0:,.2f}", (S0, 0), xycoords=("data", "axes fraction"),
                 xytext=(4, 4), textcoords="offset points", color=TEXT_2, fontsize=9)

    fig.suptitle(f"{data['ticker']} options expiring {data['expiry']}  "
                 f"(r = {data['r']:.2%}, {data['T'] * 365:.0f} days)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker", nargs="?", default="SPY")
    parser.add_argument("--days", type=int, default=30, help="target days to expiry")
    parser.add_argument("--expiry", help="exact expiry date, YYYY-MM-DD")
    parser.add_argument("--q", type=float, default=0.0, help="dividend yield, e.g. 0.013")
    parser.add_argument("--min-price", type=float, default=0.05,
                        help="skip options cheaper than this (default 0.05)")
    parser.add_argument("--heston", action="store_true",
                        help="calibrate the Heston model to the smile and plot it")
    args = parser.parse_args()

    data = fetch_market_data(args.ticker, args.expiry, args.days)
    df = compare_to_market(data, q=args.q, min_price=args.min_price)
    if df.empty:
        raise SystemExit("No usable option prices found near the spot price.")

    print(f"{data['ticker']}  spot {data['spot']:,.2f}  expiry {data['expiry']}  "
          f"T {data['T']:.3f}y  r {data['r']:.2%}  historical vol {data['hist_vol']:.1%}\n")
    heston = None
    if args.heston:
        df, params, rmse = add_heston(df, data, q=args.q)
        heston = (params, rmse)
    table = df.assign(implied_vol=df["implied_vol"] * 100)
    if heston:
        table["heston_iv"] = df["heston_iv"] * 100
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    if (df["source"] == "last").any():
        print("\nSome prices are last trades (no live bid/ask), so they may be stale.")

    if heston:
        p, rmse = heston
        print(f"\nHeston fit: v0 {p.v0:.4f} (vol {p.v0 ** 0.5:.1%}), kappa {p.kappa:.2f}, "
              f"theta {p.theta:.4f} (vol {p.theta ** 0.5:.1%}), xi {p.xi:.2f}, rho {p.rho:.2f}; "
              f"implied-vol RMSE {rmse:.2%}")

    plot(df, data, heston=heston)
    print("\nSaved market_smile.png")


if __name__ == "__main__":
    main()
