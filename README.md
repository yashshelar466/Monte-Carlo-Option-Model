# Monte-Carlo-Option-Model

Price options by simulating many possible stock price paths and averaging the discounted payoff.

## How it works

1. **Model the stock.** Under the risk-neutral measure the price follows geometric Brownian motion,
   which can be stepped exactly:
   `S(t+dt) = S(t) * exp((r - q - σ²/2) dt + σ √dt Z)`, with `Z ~ N(0,1)`.
2. **Simulate paths.** Draw random normals for `n_paths × n_steps` and build the paths (vectorised with NumPy).
3. **Compute payoffs.** Call: `max(S_T - K, 0)`, put: `max(K - S_T, 0)`, Asian: payoff on the path average.
4. **Discount and average.** Price ≈ `exp(-rT) × mean(payoff)`; standard error = `std / √n`.
5. **Reduce variance.** Antithetic variates (`Z` and `-Z`) and a control variate (the stock itself,
   whose expected value is known) shrink the error without extra paths.
6. **Validate.** European prices are checked against the Black-Scholes closed form.
7. **Greeks.** Delta, gamma and vega by bump-and-reprice using common random numbers.

## Usage

```bash
pip install -r requirements.txt
python example.py
pytest
```

```python
from mc_option import mc_european, mc_asian, bs_price

res = mc_european(S0=100, K=105, T=1, r=0.05, sigma=0.2, option="call", seed=42)
print(res.price, res.std_error, res.conf_int)
print(bs_price(100, 105, 1, 0.05, 0.2))
```

## Layout

- `mc_option/monte_carlo.py`: path simulation, European/Asian pricing, variance reduction, Greeks
- `mc_option/black_scholes.py`: analytical benchmark
- `example.py`: sample run
- `tests/`: checks against Black-Scholes

## Ideas to extend

- Barrier or lookback options (payoffs that depend on the path, like the Asian one)
- American options with Longstaff-Schwartz regression
- Heston stochastic volatility or jump-diffusion dynamics
- Quasi-random (Sobol) numbers for faster convergence
