"""Near-optimal sequential oracle for Track A.

We use a one-step Bayes-risk lookahead, the standard tractable near-optimal
policy for sequential hypothesis testing with costs:

  - Loss of stopping now with posterior pi:   lambda * (1 - max_i pi_i)
    (expected mis-identification penalty, weight lambda).
  - Value of buying the best test b first, then acting optimally (approximated
    by stopping after one more test): cost_b + E_outcome[ lambda * (1 - max pi') ].
  - STOP iff stopping risk <= expected continue value (or budget cannot afford
    the next test).

Sweeping `lambda` (the cost-of-error weight) traces the oracle's
accuracy-vs-budget frontier, which is the reference for *sufficiency regret*.

This is labelled "near-optimal", not "optimal": the exact policy is a POMDP.
The one-step lookahead is a well-established, defensible reference.
"""
from __future__ import annotations

import numpy as np

from ..core.interface import Environment, Trajectory
from ..agents.selection import best_test, expected_info_gain  # noqa: F401


def _posterior_after(posterior: np.ndarray, p_col: np.ndarray, outcome: int) -> np.ndarray:
    like = p_col if outcome == 1 else (1.0 - p_col)
    w = posterior * np.clip(like, 1e-300, None)
    return w / w.sum()


def _stop_risk(posterior: np.ndarray, lam: float) -> float:
    return lam * (1.0 - posterior.max())


def _expected_continue_value(posterior, p, costs, j, lam) -> float:
    """cost_j + E_outcome[ stop_risk(posterior') ]  (one-step lookahead)."""
    p_col = p[:, j]
    pr1 = float(posterior @ p_col)            # P(outcome=1)
    post1 = _posterior_after(posterior, p_col, 1)
    post0 = _posterior_after(posterior, p_col, 0)
    exp_future = pr1 * _stop_risk(post1, lam) + (1 - pr1) * _stop_risk(post0, lam)
    return float(costs[j]) + exp_future


def bayes_risk_oracle(lam: float):
    """Return an agent (env -> Trajectory) implementing the Bayes-risk oracle
    for cost-of-error weight `lam`."""

    def agent(env: Environment) -> Trajectory:
        # The oracle is allowed to read the environment's generative parameters
        # (that is what makes it an oracle / reference policy).
        p = env.p
        costs = env.costs
        while True:
            post = env.posterior()
            obs = env.observe()
            # Affordability: cheapest test we could still buy.
            affordable = [j for j in range(env.m) if costs[j] <= obs.budget_left + 1e-9]
            if not affordable:
                break
            j = best_test(post, p, costs)
            if costs[j] > obs.budget_left + 1e-9:
                # best test unaffordable; fall back to cheapest affordable
                j = min(affordable, key=lambda a: costs[a])
            stop_now = _stop_risk(post, lam)
            cont = _expected_continue_value(post, p, costs, j, lam)
            if stop_now <= cont:
                break
            env.query(j)
        post = env.posterior()
        ans = int(np.argmax(post))
        conf = float(post.max())
        return env.report(ans, conf, rationale=f"oracle(lambda={lam:.2f})")

    return agent


def oracle_frontier(suite_factory, lambdas):
    """Run the oracle over a fresh copy of the suite for each lambda and return
    a list of (avg_cost, avg_accuracy) points defining the reference frontier.

    `suite_factory` is a zero-arg callable returning a fresh seeded suite so that
    each lambda sees identical instances (fair comparison).
    """
    points = []
    for lam in lambdas:
        suite = suite_factory()
        agent = bayes_risk_oracle(lam)
        costs, correct = [], []
        for env in suite:
            t = agent(env)
            costs.append(t.total_cost)
            correct.append(t.correct)
        points.append((float(np.mean(costs)), float(np.mean(correct)), float(lam)))
    return points
