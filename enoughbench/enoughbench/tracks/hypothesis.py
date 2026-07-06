"""Track A -- abstract sequential hypothesis testing.

There are `k` hypotheses; one is latent ground truth (drawn from a prior).
There are `m` tests (actions). Each test j, under hypothesis i, returns a
Bernoulli(p[i, j]) outcome. Tests are repeatable: each query draws a fresh
i.i.d. outcome and costs `cost[j]`. The agent maintains a posterior over
hypotheses, decides which tests to buy and when to stop, then reports
(answer = a hypothesis id, confidence = stated probability it is correct).

This is deliberately the cleanest setting that still poses a real
"when do I have enough evidence?" decision, and it has an exact Bayesian
posterior, so the oracle is principled.

The same structure reskins to a concrete diagnostic story (hypotheses=faults,
tests=diagnostics) with no change to the math.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.interface import Environment, Evidence, Observation, StepRecord, Trajectory


class HypothesisTestingEnv(Environment):
    def __init__(
        self,
        p: np.ndarray,          # shape (k, m): P(outcome=1 | hypothesis i, test j)
        costs: np.ndarray,      # shape (m,)
        prior: np.ndarray,      # shape (k,)
        true_hyp: int,
        budget: float,
        seed,                   # int or SeedSequence: this instance's OWN outcome stream
        instance_id: int = 0,
    ):
        self.p = p
        self.costs = costs
        self.prior = prior
        self.true_hyp = int(true_hyp)
        self.budget = float(budget)
        self.instance_id = instance_id
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.k, self.m = p.shape
        self._reset_state()

    def _reset_state(self):
        # Re-seed so re-running the same instance is fully reproducible and
        # independent of any other instance or agent.
        self.rng = np.random.default_rng(self._seed)
        self._log_post = np.log(self.prior + 1e-300)   # unnormalized log-posterior
        self._spent = 0.0
        self._step = 0
        self._traj = Trajectory(instance_id=self.instance_id, ground_truth=self.true_hyp)

    # --- posterior helpers -------------------------------------------------
    def posterior(self) -> np.ndarray:
        m = self._log_post - self._log_post.max()
        w = np.exp(m)
        return w / w.sum()

    # --- Environment API ---------------------------------------------------
    def reset(self) -> Observation:
        self._reset_state()
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            question=f"Identify the true hypothesis among {self.k} candidates.",
            actions=list(range(self.m)),
            action_costs={j: float(self.costs[j]) for j in range(self.m)},
            budget_left=self.budget - self._spent,
            posterior=self.posterior().tolist(),
            step=self._step,
        )

    def budget_left(self) -> float:
        return self.budget - self._spent

    def query(self, action: int) -> Evidence:
        j = int(action)
        cost = float(self.costs[j])
        # Draw outcome from the TRUE hypothesis' model for this test.
        outcome = int(self.rng.random() < self.p[self.true_hyp, j])
        # Bayesian update: add log-likelihood of this outcome under each hypothesis.
        pj = self.p[:, j]
        ll = np.where(outcome == 1, np.log(pj + 1e-300), np.log(1.0 - pj + 1e-300))
        self._log_post = self._log_post + ll
        self._spent += cost
        self._step += 1
        self._traj.steps.append(
            StepRecord(self._step, "query", j, outcome, cost,
                       self.budget - self._spent, self.posterior().tolist())
        )
        return Evidence(action=j, outcome=outcome, cost=cost)

    def report(self, answer: int, confidence: float, rationale: str = "") -> Trajectory:
        self._step += 1
        self._traj.steps.append(
            StepRecord(self._step, "report", None, answer, 0.0,
                       self.budget - self._spent, self.posterior().tolist())
        )
        self._traj.answer = int(answer)
        self._traj.confidence = float(confidence)
        self._traj.rationale = rationale
        self._traj.total_cost = self._spent
        self._traj.correct = bool(int(answer) == self.true_hyp)
        return self._traj


def generate_instance(
    param_rng: np.random.Generator,
    seed,
    k: int = 4,
    m: int = 6,
    budget: float = 12.0,
    instance_id: int = 0,
    min_cost: float = 1.0,
    max_cost: float = 1.0,
) -> HypothesisTestingEnv:
    """Generate one seeded Track-A instance.

    `param_rng` draws the instance parameters; `seed` is this instance's OWN
    independent outcome stream (so query outcomes are reproducible and not
    coupled to other instances or to how many tests an agent buys).
    """
    p = param_rng.uniform(0.15, 0.85, size=(k, m))
    costs = param_rng.uniform(min_cost, max_cost, size=m) if max_cost > min_cost \
        else np.full(m, min_cost)
    prior = np.full(k, 1.0 / k)
    true_hyp = int(param_rng.integers(k))
    return HypothesisTestingEnv(
        p=p, costs=costs, prior=prior, true_hyp=true_hyp,
        budget=budget, seed=seed, instance_id=instance_id,
    )


def generate_suite(seed: int, n: int = 200, **kwargs) -> List[HypothesisTestingEnv]:
    """A frozen, seeded test set. Each instance gets independent param and
    outcome seeds, so results are reproducible and instance outcomes are
    mutually independent regardless of agent behavior."""
    ss = np.random.SeedSequence(seed)
    param_seeds = ss.spawn(n)
    outcome_seeds = ss.spawn(n)
    envs = []
    for i in range(n):
        prng = np.random.default_rng(param_seeds[i])
        envs.append(generate_instance(prng, seed=outcome_seeds[i],
                                       instance_id=i, **kwargs))
    return envs
