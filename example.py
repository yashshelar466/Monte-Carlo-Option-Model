"""Run the model on a sample option and compare against Black-Scholes."""

from mc_option import (HestonParams, binomial_american, bs_price, heston_price, lsm_american,
                       mc_asian, mc_european, mc_greeks, mc_heston)

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

for option in ("call", "put"):
    plain = mc_european(S0, K, T, r, sigma, option, seed=42, antithetic=False, control_variate=False)
    reduced = mc_european(S0, K, T, r, sigma, option, seed=42)
    print(f"European {option}")
    print(f"  Black-Scholes          : {bs_price(S0, K, T, r, sigma, option):.4f}")
    print(f"  MC (plain)             : {plain.price:.4f}  ± {plain.std_error:.4f}")
    print(f"  MC (antithetic + CV)   : {reduced.price:.4f}  ± {reduced.std_error:.4f}")

asian = mc_asian(S0, K, T, r, sigma, "call", seed=42)
lo, hi = asian.conf_int
print(f"\nAsian call (daily avg)   : {asian.price:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

print("\nGreeks (call):", {k: round(v, 4) for k, v in mc_greeks(S0, K, T, r, sigma).items()})

# American put: can be exercised early, priced with Longstaff-Schwartz.

S0, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
lsm = lsm_american(S0, K, T, r, sigma, "put", seed=42)
print(f"\nAmerican put (S={S0:g}, K={K:g})")
print(f"  Longstaff-Schwartz MC   : {lsm.price:.4f}  ± {lsm.std_error:.4f}")
print(f"  Binomial tree           : {binomial_american(S0, K, T, r, sigma, 'put'):.4f}")
print(f"  European (no early ex.) : {bs_price(S0, K, T, r, sigma, 'put'):.4f}")

# Heston: volatility moves randomly and rises when the stock falls (rho < 0).

params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
mc = mc_heston(100, 100, 0.5, 0.03, params, "call", seed=42)
print("\nHeston call (S=100, K=100, T=0.5)")
print(f"  Fourier formula         : {heston_price(100, 100, 0.5, 0.03, params):.4f}")
print(f"  Monte Carlo             : {mc.price:.4f}  ± {mc.std_error:.4f}")
