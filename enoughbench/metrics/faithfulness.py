"""Explanation-faithfulness (the explainable-AI dimension).

When an agent stops, it emits a short rationale. Faithfulness asks: is that
stated reason consistent with the agent's ACTUAL evidence state at stop time?

This applies to agents that produce natural-language rationales -- i.e. the LLM
baselines (Phase 2). For the Phase-1 classical baselines there is no free-text
rationale to audit, so this module is intentionally a scaffold, not a faked
score. It exposes the scoring interface and a concrete, checkable sub-signal
(claim-consistency) that does not require an LLM judge.

Two complementary signals are planned:
  1. claim_consistency (implemented): if the rationale asserts "sufficient /
     confident", the oracle posterior at stop time should indeed be confident;
     if it asserts "uncertain", it should not be. A cheap, judge-free check.
  2. reason_grounding (Phase 2, LLM-judge): do the specific reasons cited match
     the tests the agent actually ran and their outcomes?
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np

from ..core.interface import Trajectory

_CONFIDENT_RE = re.compile(r"\b(sufficient|confident|certain|enough evidence)\b", re.I)
_UNCERTAIN_RE = re.compile(r"\b(uncertain|unsure|insufficient|not enough|ambiguous)\b", re.I)


def _final_posterior(t: Trajectory) -> Optional[np.ndarray]:
    for s in reversed(t.steps):
        if s.posterior is not None:
            return np.asarray(s.posterior, dtype=float)
    return None


def claim_consistency(t: Trajectory, confident_threshold: float = 0.8) -> Optional[float]:
    """Return 1.0 if the rationale's confidence claim matches the true posterior
    state, 0.0 if it contradicts it, or None if the rationale makes no claim /
    there is no posterior to check against."""
    if not t.rationale:
        return None
    post = _final_posterior(t)
    if post is None:
        return None
    is_confident = post.max() >= confident_threshold
    says_confident = bool(_CONFIDENT_RE.search(t.rationale))
    says_uncertain = bool(_UNCERTAIN_RE.search(t.rationale))
    if not (says_confident or says_uncertain):
        return None
    if says_confident and not says_uncertain:
        return 1.0 if is_confident else 0.0
    if says_uncertain and not says_confident:
        return 1.0 if not is_confident else 0.0
    return 0.0  # contradictory rationale


def faithfulness_score(trajs: Sequence[Trajectory]) -> Optional[float]:
    """Mean claim-consistency over trajectories that make an auditable claim.
    Returns None if no trajectory provides an auditable rationale (e.g. the
    Phase-1 classical baselines)."""
    vals = [v for v in (claim_consistency(t) for t in trajs) if v is not None]
    return float(np.mean(vals)) if vals else None
