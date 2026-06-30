"""Non-LLM baseline policies for Track A.

All share the same greedy info-per-cost SELECTION; they differ ONLY in the
STOPPING rule. That isolation is the point of the benchmark.

  - max_budget   : never stop voluntarily (spend until you cannot afford a test)
  - fixed_n      : stop after n tests
  - threshold    : stop when max posterior >= tau   (a simple confidence rule)
  - random       : random test choice + random stop (weak floor)
"""
from __future__ import annotations

import numpy as np

from ..core.interface import Environment, Trajectory
from .selection import best_test


def _finish(env: Environment) -> Trajectory:
    post = env.posterior()
    return env.report(int(np.argmax(post)), float(post.max()))


def max_budget_agent(env: Environment) -> Trajectory:
    while True:
        obs = env.observe()
        affordable = [j for j in range(env.m) if env.costs[j] <= obs.budget_left + 1e-9]
        if not affordable:
            break
        j = best_test(env.posterior(), env.p, env.costs)
        if env.costs[j] > obs.budget_left + 1e-9:
            j = min(affordable, key=lambda a: env.costs[a])
        env.query(j)
    return _finish(env)


def fixed_n_agent(n: int):
    def agent(env: Environment) -> Trajectory:
        for _ in range(n):
            obs = env.observe()
            affordable = [j for j in range(env.m) if env.costs[j] <= obs.budget_left + 1e-9]
            if not affordable:
                break
            j = best_test(env.posterior(), env.p, env.costs)
            if env.costs[j] > obs.budget_left + 1e-9:
                j = min(affordable, key=lambda a: env.costs[a])
            env.query(j)
        return _finish(env)
    return agent


def threshold_agent(tau: float):
    def agent(env: Environment) -> Trajectory:
        while True:
            post = env.posterior()
            if post.max() >= tau:
                break
            obs = env.observe()
            affordable = [j for j in range(env.m) if env.costs[j] <= obs.budget_left + 1e-9]
            if not affordable:
                break
            j = best_test(post, env.p, env.costs)
            if env.costs[j] > obs.budget_left + 1e-9:
                j = min(affordable, key=lambda a: env.costs[a])
            env.query(j)
        return _finish(env)
    return agent


def random_agent(stop_prob: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)

    def agent(env: Environment) -> Trajectory:
        while True:
            obs = env.observe()
            affordable = [j for j in range(env.m) if env.costs[j] <= obs.budget_left + 1e-9]
            if not affordable or rng.random() < stop_prob:
                break
            env.query(int(rng.choice(affordable)))
        return _finish(env)
    return agent
