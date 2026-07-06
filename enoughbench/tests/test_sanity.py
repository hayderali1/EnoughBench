"""Sanity tests -- run with: python -m pytest -q  (or python tests/test_sanity.py)"""
import numpy as np

from enoughbench.tracks.hypothesis import generate_suite, generate_instance
from enoughbench.oracles.bayes_sequential import bayes_risk_oracle, oracle_frontier
from enoughbench.agents.baselines import max_budget_agent, fixed_n_agent
from enoughbench.agents.selection import expected_info_gain
from enoughbench.metrics.metrics import accuracy, avg_cost


def test_posterior_normalized_and_updates():
    rng = np.random.default_rng(0)
    env = generate_instance(rng, seed=0, k=4, m=6, budget=20)
    env.reset()
    p0 = env.posterior()
    assert abs(p0.sum() - 1.0) < 1e-9
    for _ in range(5):
        env.query(0)
        p = env.posterior()
        assert abs(p.sum() - 1.0) < 1e-9
        assert np.all(p >= 0)


def test_more_budget_helps_accuracy():
    """max_budget should be at least as accurate as a tiny fixed budget."""
    frontier = oracle_frontier(lambda: generate_suite(1, n=150), [1, 5, 20, 100])
    acc_max = accuracy([max_budget_agent(e) for e in generate_suite(1, n=150)])
    acc_few = accuracy([fixed_n_agent(1)(e) for e in generate_suite(1, n=150)])
    assert acc_max >= acc_few - 1e-9


def test_info_gain_nonnegative():
    rng = np.random.default_rng(3)
    env = generate_instance(rng, seed=0, k=5, m=8, budget=20)
    ig = expected_info_gain(env.posterior(), env.p)
    assert np.all(ig >= -1e-9)


def test_oracle_lambda_monotonicity():
    """Higher error-penalty lambda should not REDUCE average cost spent."""
    suite_f = lambda: generate_suite(2, n=150)
    costs = []
    for lam in [1, 5, 20, 100]:
        agent = bayes_risk_oracle(lam)
        costs.append(avg_cost([agent(e) for e in suite_f()]))
    # weakly increasing
    for a, b in zip(costs, costs[1:]):
        assert b >= a - 0.25  # small tolerance for ties/noise


if __name__ == "__main__":
    test_posterior_normalized_and_updates()
    test_more_budget_helps_accuracy()
    test_info_gain_nonnegative()
    test_oracle_lambda_monotonicity()
    print("all sanity tests passed")
