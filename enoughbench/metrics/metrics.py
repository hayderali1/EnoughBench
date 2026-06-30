"""Metrics for EnoughBench.

  - accuracy / avg_cost : task quality and effort
  - ECE                 : calibration of stated confidence at stop time
  - sufficiency_regret  : excess cost vs the oracle frontier at equal accuracy
                          (the headline, nameable metric)

Explanation-faithfulness lives in `faithfulness.py` (applies to agents that
emit a rationale, i.e. the LLM baselines in Phase 2).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..core.interface import Trajectory


def accuracy(trajs: Sequence[Trajectory]) -> float:
    return float(np.mean([t.correct for t in trajs]))


def avg_cost(trajs: Sequence[Trajectory]) -> float:
    return float(np.mean([t.total_cost for t in trajs]))


def expected_calibration_error(trajs: Sequence[Trajectory], n_bins: int = 10) -> float:
    """Standard ECE over reported confidence vs empirical correctness."""
    conf = np.array([t.confidence for t in trajs], dtype=float)
    correct = np.array([1.0 if t.correct else 0.0 for t in trajs], dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(trajs)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _oracle_cost_at_accuracy(frontier: List[Tuple[float, float, float]], acc: float) -> float:
    """Interpolate the oracle's cost at a target accuracy.

    frontier: list of (cost, accuracy, lambda), not assumed sorted.
    Returns the minimal oracle cost achieving >= acc, with linear interpolation
    between the two bracketing frontier points.
    """
    pts = sorted([(a, c) for (c, a, _) in frontier])  # by accuracy
    accs = [a for a, _ in pts]
    costs = [c for _, c in pts]
    if acc <= accs[0]:
        return costs[0]
    if acc >= accs[-1]:
        return costs[-1]
    for i in range(1, len(accs)):
        if accs[i] >= acc:
            a0, a1 = accs[i - 1], accs[i]
            c0, c1 = costs[i - 1], costs[i]
            if a1 == a0:
                return min(c0, c1)
            w = (acc - a0) / (a1 - a0)
            return c0 + w * (c1 - c0)
    return costs[-1]


def sufficiency_regret(
    trajs: Sequence[Trajectory],
    frontier: List[Tuple[float, float, float]],
) -> float:
    """Excess cost the agent paid relative to the oracle frontier at the SAME
    accuracy. >= 0; lower is better. If the agent is below the frontier's lowest
    accuracy, regret is reported as the agent's cost minus the cheapest oracle
    point (it spent budget yet under-performed the cheapest reference)."""
    agent_acc = accuracy(trajs)
    agent_cost = avg_cost(trajs)
    oracle_cost = _oracle_cost_at_accuracy(frontier, agent_acc)
    return float(max(agent_cost - oracle_cost, 0.0))


def summarize(name: str, trajs: Sequence[Trajectory],
              frontier: List[Tuple[float, float, float]]) -> dict:
    return {
        "agent": name,
        "accuracy": round(accuracy(trajs), 4),
        "avg_cost": round(avg_cost(trajs), 4),
        "ece": round(expected_calibration_error(trajs), 4),
        "sufficiency_regret": round(sufficiency_regret(trajs, frontier), 4),
    }
