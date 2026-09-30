import pytest

from mc_option import binomial_american, bs_price, lsm_american

K, r, sigma = 40.0, 0.06, 0.2


@pytest.mark.parametrize("S0,T", [(36, 1), (40, 1), (44, 2)])
def test_lsm_put_matches_binomial_tree(S0, T):
    tree = binomial_american(S0, K, T, r, sigma, "put")
    res = lsm_american(S0, K, T, r, sigma, "put", n_paths=50_000, n_steps=50 * T, seed=0)
    # LSM exercises on 50 dates a year (Bermudan), so it may sit slightly
    # below the continuously exercisable tree value.
    assert tree - 0.03 < res.price < tree + 4 * res.std_error


def test_binomial_matches_published_value():
    # Longstaff-Schwartz (2001), Table 1 case: well-known American put value.
    assert binomial_american(36, K, 1, r, sigma, "put") == pytest.approx(4.487, abs=0.002)


def test_early_exercise_premium_for_puts():
    american = lsm_american(36, K, 1, r, sigma, "put", n_paths=50_000, seed=1).price
    assert american > bs_price(36, K, 1, r, sigma, "put") + 0.5


def test_call_without_dividends_is_never_exercised_early():
    res = lsm_american(100, 105, 1, 0.05, 0.2, "call", n_paths=50_000, seed=2)
    european = bs_price(100, 105, 1, 0.05, 0.2, "call")
    assert abs(res.price - european) < 4 * res.std_error
    assert binomial_american(100, 105, 1, 0.05, 0.2, "call") == pytest.approx(european, abs=0.005)


def test_dividends_make_early_exercise_worth_it_for_calls():
    american = binomial_american(100, 100, 1, 0.03, 0.2, "call", q=0.07)
    lsm = lsm_american(100, 100, 1, 0.03, 0.2, "call", q=0.07, n_paths=50_000, seed=3)
    assert american > bs_price(100, 100, 1, 0.03, 0.2, "call", q=0.07) + 0.3
    assert abs(lsm.price - american) < 0.03 + 4 * lsm.std_error
