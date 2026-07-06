"""Track C baselines (change-detection).

Shared selection: most-discriminative slot per cost. Differ only in stopping.
The answer is binary (changed / not); confidence = max(P(change), 1-P(change)).
"""
from __future__ import annotations

import numpy as np

from ..core.interface import Environment, Trajectory
from ..tracks.change_detection import best_slot


def _affordable(env: Environment):
    bl = env.budget_left()
    return [a for a in range(env.m) if env.costs[a] <= bl + 1e-9]


def _finish(env: Environment) -> Trajectory:
    pc = env.p_change()
    ans = 1 if pc >= 0.5 else 0
    conf = max(pc, 1.0 - pc)
    return env.report(ans, float(conf))


def _step_query(env: Environment) -> bool:
    aff = _affordable(env)
    if not aff:
        return False
    j = best_slot(env)
    if env.costs[j] > env.budget_left() + 1e-9:
        j = min(aff, key=lambda a: env.costs[a])
    env.query(j)
    return True


def max_budget_agent(env: Environment) -> Trajectory:
    while _step_query(env):
        pass
    return _finish(env)


def fixed_n_agent(n: int):
    def agent(env: Environment) -> Trajectory:
        for _ in range(n):
            if not _step_query(env):
                break
        return _finish(env)
    return agent


def threshold_agent(tau: float):
    def agent(env: Environment) -> Trajectory:
        while True:
            pc = env.p_change()
            if max(pc, 1.0 - pc) >= tau:
                break
            if not _step_query(env):
                break
        return _finish(env)
    return agent


def random_agent(stop_prob: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)

    def agent(env: Environment) -> Trajectory:
        while True:
            aff = _affordable(env)
            if not aff or rng.random() < stop_prob:
                break
            env.query(int(rng.choice(aff)))
        return _finish(env)
    return agent
