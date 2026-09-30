"""Plot how the Monte Carlo price converges to Black-Scholes as paths increase.

Saves convergence.png. Run:  python convergence_chart.py
"""

import matplotlib.pyplot as plt
import numpy as np

from mc_option import convergence_study

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

# Chart palette (validated for colour-blind separation and contrast).
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PLAIN, REDUCED, REFERENCE = "#eb6834", "#2a78d6", "#52514e"

data = convergence_study(S0, K, T, r, sigma, "call")
n = data["n_paths"]
bs = data["bs_price"]

plt.rcParams.update({
    "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
    "axes.edgecolor": GRID, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Number of simulated paths")

# Left: price estimate with 95% confidence band, converging on Black-Scholes.
ax1.axhline(bs, color=REFERENCE, linestyle="--", linewidth=1.2,
            label=f"Black-Scholes {bs:.3f}")
for key, color, label in (("plain", PLAIN, "Plain MC"),
                          ("reduced", REDUCED, "Antithetic + control variate")):
    price, se = data[f"{key}_price"], data[f"{key}_se"]
    ax1.fill_between(n, price - 1.96 * se, price + 1.96 * se, color=color,
                     alpha=0.15, linewidth=0)
    ax1.plot(n, price, color=color, linewidth=2, marker="o", markersize=4,
             markeredgecolor=SURFACE, markeredgewidth=1, label=label)
ax1.set_ylabel("Estimated call price")
ax1.set_title("Price estimate and 95% confidence band", loc="left",
              fontsize=11, color=TEXT)
ax1.legend(frameon=False, loc="upper right")

# Right: standard error on log-log axes, where 1/sqrt(N) is a straight line.
ax2.set_yscale("log")
for key, color, label in (("plain", PLAIN, "Plain MC"),
                          ("reduced", REDUCED, "Antithetic + control variate")):
    se = data[f"{key}_se"]
    ax2.plot(n, se, color=color, linewidth=2, marker="o", markersize=4,
             markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    ax2.annotate(f"{se[-1]:.4f}", (n[-1], se[-1]), xytext=(6, 0),
                 textcoords="offset points", va="center", color=TEXT_2, fontsize=9)
ref = data["plain_se"][0] * np.sqrt(n[0] / n)
ax2.plot(n, ref, color=REFERENCE, linestyle="--", linewidth=1.2,
         label=r"$1/\sqrt{N}$ reference")
ratio = np.median(data["plain_se"] / data["reduced_se"])
ax2.set_ylabel("Standard error")
ax2.set_title(f"Standard error falls as 1/√N; variance reduction is ~{ratio:.1f}× tighter",
              loc="left", fontsize=11, color=TEXT)
ax2.legend(frameon=False, loc="lower left")
ax2.margins(x=0.08)

fig.suptitle(f"Monte Carlo convergence: European call  (S={S0:g}, K={K:g}, T={T:g}, "
             f"r={r:g}, σ={sigma:g})", x=0.01, ha="left", fontsize=12, color=TEXT)
fig.tight_layout()
fig.savefig("convergence.png", dpi=150, facecolor=SURFACE)
print(f"Saved convergence.png (Black-Scholes {bs:.4f}; "
      f"final estimates plain {data['plain_price'][-1]:.4f}, "
      f"reduced {data['reduced_price'][-1]:.4f}; SE ratio ~{ratio:.1f}x)")
