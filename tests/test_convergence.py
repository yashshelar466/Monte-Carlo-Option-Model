import numpy as np

from mc_option import convergence_study

S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2


def test_convergence_study():
    n = [1_000, 10_000, 100_000]
    data = convergence_study(S0, K, T, r, sigma, path_counts=n)
    assert list(data["n_paths"]) == n
    for key in ("plain", "reduced"):
        se = data[f"{key}_se"]
        # Error shrinks like 1/sqrt(N): 100x the paths gives ~10x smaller error.
        assert np.all(np.diff(se) < 0)
        assert 7 < se[0] / se[-1] < 14
        assert np.all(np.abs(data[f"{key}_price"] - data["bs_price"]) < 4 * se)
    assert np.all(data["reduced_se"] < data["plain_se"])
