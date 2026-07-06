"""Track C -- change-detection / temporal sufficiency.

A scalar signal is observed over T discrete time slots. With probability 0.5 a
change occurs at an unknown time tau (the mean jumps from mu0 to mu1); otherwise
no change occurs. The agent actively samples slots (each sample is noisy and
costs budget; slots are repeatable) and must decide WHEN it has enough evidence
to declare "a change occurred" or "no change", with a calibrated confidence.

This adds a *temporal* sufficiency decision absent from Tracks A/B and maps onto
online-monitoring / change-point-detection. The belief is an exact Bayesian
posterior over {no-change, change-at-tau} hypotheses, so the standard metric
suite (accuracy, ECE, sufficiency regret) applies, and a Bayes-optimal threshold
oracle defines the frontier.

Hypothesis layout (K = T):
  h = 0           -> no change
  h = 1..T-1      -> change at time tau = h   (post-change for slots t >= tau)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.interface import Environment, Evidence, Observation, StepRecord, Trajectory


class ChangeDetectionEnv(Environment):
    def __init__(self, T: int, mu0: float, mu1: float, sigma: float,
                 prior: np.ndarray, true_hyp: int, budget: float, seed,
                 cost: float = 1.0, instance_id: int = 0):
        self.T = T
        self.mu0, self.mu1, self.sigma = mu0, mu1, sigma
        self.prior = prior                    # over K=T hypotheses
        self.true_hyp = int(true_hyp)         # 0 = no change; tau otherwise
        self.budget = float(budget)
        self.instance_id = instance_id
        self._seed = seed
        self.k = T                            # number of hypotheses
        self.m = T                            # actions = time slots
        self.costs = np.full(T, cost)
        # means[h, t] = mean of slot t under hypothesis h
        self.means = np.full((T, T), mu0, dtype=float)
        for tau in range(1, T):
            self.means[tau, tau:] = mu1       # change at tau affects slots >= tau
        self.changed = int(true_hyp != 0)
        self._reset_state()

    def _reset_state(self):
        self.rng = np.random.default_rng(self._seed)
        self._log_post = np.log(self.prior + 1e-300)
        self._spent = 0.0
        self._step = 0
        self._traj = Trajectory(instance_id=self.instance_id, ground_truth=self.changed)

    # --- belief ------------------------------------------------------------
    def posterior(self) -> np.ndarray:
        m = self._log_post - self._log_post.max()
        w = np.exp(m)
        return w / w.sum()

    def p_change(self) -> float:
        return float(1.0 - self.posterior()[0])

    # --- API ---------------------------------------------------------------
    def reset(self) -> Observation:
        self._reset_state()
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            question=f"Did a change occur over {self.T} time slots? Sample slots, then report.",
            actions=list(range(self.T)),
            action_costs={t: float(self.costs[t]) for t in range(self.T)},
            budget_left=self.budget - self._spent,
            posterior=self.posterior().tolist(),
            step=self._step,
        )

    def budget_left(self) -> float:
        return self.budget - self._spent

    def query(self, action: int) -> Evidence:
        t = int(action)
        cost = float(self.costs[t])
        y = float(self.means[self.true_hyp, t] + self.rng.normal(0, self.sigma))
        # Gaussian log-likelihood of y at slot t under every hypothesis.
        ll = -0.5 * ((y - self.means[:, t]) / self.sigma) ** 2
        self._log_post = self._log_post + ll
        self._spent += cost
        self._step += 1
        self._traj.steps.append(
            StepRecord(self._step, "query", t, round(y, 4), cost,
                       self.budget - self._spent, self.posterior().tolist()))
        return Evidence(action=t, outcome=y, cost=cost)

    def report(self, answer: int, confidence: float, rationale: str = "") -> Trajectory:
        self._step += 1
        self._traj.steps.append(
            StepRecord(self._step, "report", None, int(answer), 0.0,
                       self.budget - self._spent, self.posterior().tolist()))
        self._traj.answer = int(answer)
        self._traj.confidence = float(confidence)
        self._traj.rationale = rationale
        self._traj.total_cost = self._spent
        self._traj.correct = bool(int(answer) == self.changed)
        return self._traj


def slot_info_scores(posterior: np.ndarray, means: np.ndarray,
                     sigma: float, costs: np.ndarray) -> np.ndarray:
    """Discriminative score per slot: posterior-weighted variance of predicted
    means at that slot (high where surviving hypotheses disagree), per unit cost.
    Cheap, sigma-aware proxy for expected information gain."""
    mbar = posterior @ means                     # (T,) expected mean per slot
    var = posterior @ (means ** 2) - mbar ** 2    # (T,) weighted variance
    return np.clip(var, 0, None) / (sigma ** 2) / np.maximum(costs, 1e-9)


def best_slot(env: ChangeDetectionEnv) -> int:
    return int(np.argmax(slot_info_scores(env.posterior(), env.means,
                                          env.sigma, env.costs)))


def generate_instance(param_rng, seed, T=10, mu0=0.0, delta=1.5, sigma=1.0,
                      budget=12.0, p_change=0.5, cost=1.0, instance_id=0):
    changed = param_rng.random() < p_change
    if changed:
        true_hyp = int(param_rng.integers(1, T))   # tau in 1..T-1
    else:
        true_hyp = 0
    prior = np.empty(T)
    prior[0] = 1.0 - p_change
    prior[1:] = p_change / (T - 1)
    return ChangeDetectionEnv(T=T, mu0=mu0, mu1=mu0 + delta, sigma=sigma,
                              prior=prior, true_hyp=true_hyp, budget=budget,
                              seed=seed, cost=cost, instance_id=instance_id)


def generate_suite(seed: int, n: int = 200, **kw) -> List[ChangeDetectionEnv]:
    ss = np.random.SeedSequence(seed)
    pseeds, oseeds = ss.spawn(n), ss.spawn(n)
    return [generate_instance(np.random.default_rng(pseeds[i]), seed=oseeds[i],
                              instance_id=i, **kw) for i in range(n)]
