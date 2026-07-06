"""Track B baseline policies.

All share GP-UCB-per-cost SELECTION (where to measure) and differ ONLY in the
STOPPING rule, isolating the sufficiency decision -- as in Track A.

  - max_budget : measure until you cannot afford any action
  - fixed_n    : measure n times
  - threshold  : stop when P(argmax at best cell) >= tau
  - random     : random location + random stop (weak floor)
"""
from __future__ import annotations

import numpy as np

from ..core.interface import Environment, Trajectory


def _affordable(env: Environment, budget_left: float):
    return [a for a in range(env.A) if env.action_costs[a] <= budget_left + 1e-9]


def _best_action(env: Environment, affordable):
    scores = env.ucb_scores()
    aff = np.array(affordable)
    return int(aff[np.argmax(scores[aff])])


def _finish(env: Environment) -> Trajectory:
    best, conf, _ = env.localize()
    return env.report(int(best), float(conf))


def max_budget_agent(env: Environment) -> Trajectory:
    while True:
        aff = _affordable(env, env.budget_left())
        if not aff:
            break
        env.query(_best_action(env, aff))
    return _finish(env)


def fixed_n_agent(n: int):
    def agent(env: Environment) -> Trajectory:
        for _ in range(n):
            aff = _affordable(env, env.budget_left())
            if not aff:
                break
            env.query(_best_action(env, aff))
        return _finish(env)
    return agent


def threshold_agent(tau: float):
    def agent(env: Environment) -> Trajectory:
        while True:
            best, conf, _ = env.localize()
            if float(conf) >= tau:
                break
            aff = _affordable(env, env.budget_left())
            if not aff:
                break
            env.query(_best_action(env, aff))
        return _finish(env)
    return agent


def random_agent(stop_prob: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)

    def agent(env: Environment) -> Trajectory:
        while True:
            aff = _affordable(env, env.budget_left())
            if not aff or rng.random() < stop_prob:
                break
            env.query(int(rng.choice(aff)))
        return _finish(env)
    return agent
