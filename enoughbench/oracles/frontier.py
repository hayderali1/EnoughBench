"""Reference (best-known) accuracy-vs-budget frontier.

Sufficiency regret must be measured against the best achievable cost/accuracy
tradeoff by strong, MODEL-AWARE policies (policies that know the generative
model). A single one-step Bayes-risk oracle is too myopic to span the whole
tradeoff, so we define the reference as the **Pareto front** over a basket of
strong policies:

  - confidence-threshold stopping (info-greedy selection) swept over tau
  - fixed-n stopping swept over n
  - Bayes-risk oracle swept over lambda

Evaluated agents that are themselves model-aware and near-optimal will sit ON
this front (regret ~ 0) -- that is the correct sanity check. The gap opens for
weak policies and, in Phase 2, for LLM agents that must reason through the API
without the true model.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from ..agents.baselines import fixed_n_agent, threshold_agent, max_budget_agent
from .bayes_sequential import bayes_risk_oracle


def _operating_point(suite_factory, agent) -> Tuple[float, float]:
    trajs = [agent(env) for env in suite_factory()]
    cost = float(np.mean([t.total_cost for t in trajs]))
    acc = float(np.mean([t.correct for t in trajs]))
    return cost, acc


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Keep points that are not dominated (sorted by cost asc, accuracy strictly
    increasing)."""
    pts = sorted(points)
    front, best_acc = [], -np.inf
    for c, a in pts:
        if a > best_acc + 1e-12:
            front.append((c, a))
            best_acc = a
    return front


def reference_frontier(
    suite_factory: Callable,
    taus=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999),
    fixed_ns=(0, 1, 2, 3, 4, 6, 8, 10, 12),
    lambdas=(1, 3, 8, 20, 60, 150),
) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float]] = []
    for tau in taus:
        pts.append(_operating_point(suite_factory, threshold_agent(tau)))
    for n in fixed_ns:
        pts.append(_operating_point(suite_factory, fixed_n_agent(n)))
    for lam in lambdas:
        pts.append(_operating_point(suite_factory, bayes_risk_oracle(lam)))
    pts.append(_operating_point(suite_factory, max_budget_agent))
    front = pareto_front(pts)
    # Return as (cost, accuracy, tag) to match the metrics interface.
    return [(c, a, 0.0) for (c, a) in front]
