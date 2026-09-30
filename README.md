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

## American options (Longstaff-Schwartz)

An American option can be exercised any day, so its value depends on the best exercise policy.
Simulation runs forward in time, but the exercise decision needs the value of *waiting*, which lives in the future.
Longstaff-Schwartz solves this by walking backwards through the simulated dates:

1. At expiry, each path's cash flow is its payoff.
2. At each earlier date, take the paths that are in the money and regress their discounted future
   cash flow on `1, S/K, (S/K)², (S/K)³`. The fitted curve is the estimated value of waiting.
3. Where exercising now pays more than that estimate, exercise: that path's cash flow becomes today's payoff.
4. The price is the average cash flow, discounted to today.

| American put, S=36, K=40, T=1, r=6%, σ=20% | Price |
|---|---|
| Longstaff-Schwartz MC (100k paths, 50 dates) | 4.482 ± 0.006 |
| Binomial tree (2,000 steps) | 4.487 |
| European put (Black-Scholes) | 3.844 |

The 0.64 gap is the early-exercise premium. A call on a stock that pays no dividends is never worth
exercising early, so its American and European prices match; `tests/test_american.py` checks both facts.

## Convergence

![Monte Carlo convergence](convergence.png)

Left: each estimate and its 95% confidence band close in on the Black-Scholes price as paths grow.
Right: on log-log axes the standard error falls along the 1/√N line, so 100× the paths buys only 10× the accuracy.
Variance reduction shifts the whole line down (about 1.6× tighter here), which is equivalent to running
about 2.5× as many plain paths for free. Regenerate with `python convergence_chart.py`.

## Usage

```bash
pip install -r requirements.txt
python example.py
python convergence_chart.py
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
- `mc_option/american.py`: Longstaff-Schwartz American pricing and a binomial-tree benchmark
- `mc_option/black_scholes.py`: analytical benchmark
- `mc_option/convergence.py`: convergence study data
- `example.py`: sample run
- `convergence_chart.py`: draws `convergence.png`
- `tests/`: checks against Black-Scholes

## Ideas to extend

- Barrier or lookback options (payoffs that depend on the path, like the Asian one)
- Heston stochastic volatility or jump-diffusion dynamics
- Quasi-random (Sobol) numbers for faster convergence
