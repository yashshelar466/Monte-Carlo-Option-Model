"""Run the model on a sample option and compare against Black-Scholes."""

from mc_option import bs_price, mc_asian, mc_european, mc_greeks

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
