"""Shared measurement-selection logic.

Both the oracle and the LLM/classical baselines select the next test by greedy
expected information gain per unit cost. Isolating selection here means the
policies differ ONLY in their stopping rule -- which is exactly the decision
EnoughBench is built to measure.
"""
from __future__ import annotations

import numpy as np


def _binary_entropy(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def expected_info_gain(posterior: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mutual information I(H; O_j) in bits, for each test j.

    posterior: (k,)   p: (k, m) Bernoulli params.
    """
    # Marginal P(O_j = 1) = sum_i posterior_i * p[i, j]
    marg1 = posterior @ p                      # (m,)
    h_marg = _binary_entropy(marg1)            # entropy of the outcome
    # Conditional entropy E_i[ H(O_j | H=i) ] = sum_i posterior_i * Hb(p[i,j])
    h_cond = posterior @ _binary_entropy(p)    # (m,)
    return np.clip(h_marg - h_cond, 0.0, None)


def best_test(posterior: np.ndarray, p: np.ndarray, costs: np.ndarray) -> int:
    """Pick the test with the highest expected info gain per unit cost."""
    ig = expected_info_gain(posterior, p)
    score = ig / np.maximum(costs, 1e-9)
    return int(np.argmax(score))
