"""Track B -- spatial-field localization via simulated IoT sensor-network coordination.

A Gaussian-random-field over a GxG grid is the latent quantity (e.g. soil
moisture / a pollutant). A network of fixed IoT nodes plus robot-reachable
waypoints are candidate measurement locations; each query returns a noisy point
reading and costs energy (nodes cheap, robot waypoints costlier). The agent
coordinates the network -- choosing where to measure -- maintains a GP posterior,
and must report the grid cell containing the field MAXIMUM, with a confidence,
then stop. Correct iff the reported cell is within `tol` of the true argmax.

Belief over the argmax is a proper distribution: we Monte-Carlo sample the GP
posterior over the top-K candidate cells and count argmax frequencies. This
mirrors Track A's posterior-over-answers, so the same metric suite applies.

All simulated; no ROS/Gazebo/hardware.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..core.interface import Environment, Evidence, Observation, StepRecord, Trajectory


def _rbf(A: np.ndarray, B: np.ndarray, sig2: float, ell: float) -> np.ndarray:
    d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
    return sig2 * np.exp(-0.5 * np.clip(d2, 0, None) / (ell**2))


class SpatialFieldEnv(Environment):
    def __init__(
        self,
        grid: int,
        action_locs: np.ndarray,      # (A,2) candidate measurement coordinates
        action_costs: np.ndarray,     # (A,)
        f_true: np.ndarray,           # (grid*grid,) latent field over all cells
        budget: float,
        sig2: float,
        ell: float,
        noise: float,
        seed,
        tol: float = 1.5,
        topk: int = 12,
        n_samples: int = 150,
        ucb_beta: float = 2.0,
        instance_id: int = 0,
    ):
        self.G = grid
        self.cells = np.array([(i, j) for i in range(grid) for j in range(grid)],
                              dtype=float)
        self.action_locs = action_locs
        self.action_costs = action_costs
        self.f_true = f_true
        self.budget = float(budget)
        self.sig2, self.ell, self.noise = sig2, ell, noise
        self.tol = tol
        self.topk = topk
        self.n_samples = n_samples
        self.ucb_beta = ucb_beta
        self._seed = seed
        self.instance_id = instance_id
        self.true_argmax = int(np.argmax(f_true))
        self.A = len(action_locs)
        self.m = self.A
        self.costs = action_costs  # alias for harness symmetry
        self._reset_state()

    def _reset_state(self):
        ss = self._seed if isinstance(self._seed, np.random.SeedSequence) \
            else np.random.SeedSequence(self._seed)
        oc, bc = ss.spawn(2)
        self.rng = np.random.default_rng(oc)             # measurement noise only
        self._belief_rng = np.random.default_rng(bc)     # MC belief sampling only
        self._Xobs: List[np.ndarray] = []
        self._yobs: List[float] = []
        self._spent = 0.0
        self._step = 0
        self._traj = Trajectory(instance_id=self.instance_id,
                                ground_truth=self.true_argmax)

    # --- GP posterior -------------------------------------------------------
    def _posterior(self, Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, full covariance) of the GP posterior at Xstar."""
        kss = _rbf(Xstar, Xstar, self.sig2, self.ell)
        if not self._Xobs:
            return np.zeros(len(Xstar)), kss
        X = np.array(self._Xobs)
        y = np.array(self._yobs)
        Ky = _rbf(X, X, self.sig2, self.ell) + self.noise * np.eye(len(X))
        c = cho_factor(Ky + 1e-8 * np.eye(len(X)))
        Ks = _rbf(Xstar, X, self.sig2, self.ell)
        mean = Ks @ cho_solve(c, y)
        cov = kss - Ks @ cho_solve(c, Ks.T)
        return mean, cov

    def _post_mean_var_cells(self) -> Tuple[np.ndarray, np.ndarray]:
        mean, cov = self._posterior(self.cells)
        return mean, np.clip(np.diag(cov), 1e-12, None)

    def localize(self) -> Tuple[int, float, np.ndarray]:
        """Return (reported_cell, tolerance-aware confidence, exact-cell prob vec).

        Confidence is P(true argmax lies within `tol` of the reported cell) under
        the GP posterior -- i.e. it matches the event that is actually scored,
        which is what makes it calibratable and makes threshold-stopping work.
        """
        mean, _ = self._post_mean_var_cells()
        cand = np.argsort(mean)[-self.topk:]
        m_c, cov_c = self._posterior(self.cells[cand])
        L = np.linalg.cholesky(cov_c + 1e-8 * np.eye(len(cand)))
        z = self._belief_rng.standard_normal((len(cand), self.n_samples))
        draws = m_c[:, None] + L @ z                      # (K, S)
        winners = cand[np.argmax(draws, axis=0)]          # sampled argmax cells
        best = int(cand[int(np.argmax(m_c))])             # report posterior-mean argmax
        br, bc = divmod(best, self.G)
        wr, wc = winners // self.G, winners % self.G
        conf = float(np.mean(np.hypot(wr - br, wc - bc) <= self.tol))
        probs = np.zeros(self.G * self.G)
        for w in winners:
            probs[w] += 1.0 / self.n_samples
        return best, conf, probs

    def ucb_scores(self) -> np.ndarray:
        """GP-UCB-per-cost acquisition over the candidate ACTION locations."""
        mean, cov = self._posterior(self.action_locs)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        return (mean + self.ucb_beta * std) / np.maximum(self.action_costs, 1e-9)

    # --- Environment API ----------------------------------------------------
    def reset(self) -> Observation:
        self._reset_state()
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            question=f"Report the {self.G}x{self.G} grid cell containing the field maximum.",
            actions=list(range(self.A)),
            action_costs={a: float(self.action_costs[a]) for a in range(self.A)},
            budget_left=self.budget - self._spent,
            posterior=None,
            step=self._step,
        )

    def budget_left(self) -> float:
        return self.budget - self._spent

    def query(self, action: int) -> Evidence:
        a = int(action)
        loc = self.action_locs[a]
        cost = float(self.action_costs[a])
        cell_idx = int(loc[0] * self.G + loc[1])
        val = float(self.f_true[cell_idx] + self.rng.normal(0, np.sqrt(self.noise)))
        self._Xobs.append(loc.astype(float))
        self._yobs.append(val)
        self._spent += cost
        self._step += 1
        self._traj.steps.append(
            StepRecord(self._step, "query", a, round(val, 4), cost,
                       self.budget - self._spent, None)
        )
        return Evidence(action=a, outcome=val, cost=cost)

    def report(self, answer: int, confidence: float, rationale: str = "") -> Trajectory:
        self._step += 1
        _, _, probs = self.localize()
        self._traj.steps.append(
            StepRecord(self._step, "report", None, int(answer), 0.0,
                       self.budget - self._spent, probs.tolist())
        )
        rc, ac = divmod(int(answer), self.G)
        tr, tc = divmod(self.true_argmax, self.G)
        dist = np.hypot(rc - tr, ac - tc)
        self._traj.answer = int(answer)
        self._traj.confidence = float(confidence)
        self._traj.rationale = rationale
        self._traj.total_cost = self._spent
        self._traj.correct = bool(dist <= self.tol)
        self._traj.meta["loc_error"] = float(dist)
        return self._traj


def generate_instance(param_rng, seed, grid=12, n_nodes=12, budget=18.0,
                      sig2=1.0, ell=2.0, noise=0.05, node_cost=1.0,
                      robot_cost=2.0, instance_id=0, **kw) -> SpatialFieldEnv:
    cells = np.array([(i, j) for i in range(grid) for j in range(grid)], dtype=float)
    # Ground-truth field ~ GP prior.
    K = _rbf(cells, cells, sig2, ell) + 1e-6 * np.eye(len(cells))
    L = np.linalg.cholesky(K)
    f_true = L @ param_rng.standard_normal(len(cells))
    # IoT nodes: random cells (cheap). Robot waypoints: coarse subgrid (costlier).
    node_idx = param_rng.choice(len(cells), size=n_nodes, replace=False)
    node_locs = cells[node_idx]
    step = max(2, grid // 4)
    wp = np.array([(i, j) for i in range(0, grid, step) for j in range(0, grid, step)],
                  dtype=float)
    action_locs = np.vstack([node_locs, wp])
    action_costs = np.concatenate([np.full(len(node_locs), node_cost),
                                   np.full(len(wp), robot_cost)])
    return SpatialFieldEnv(
        grid=grid, action_locs=action_locs, action_costs=action_costs,
        f_true=f_true, budget=budget, sig2=sig2, ell=ell, noise=noise,
        seed=seed, instance_id=instance_id, **kw,
    )


def generate_suite(seed: int, n: int = 80, **kwargs) -> List[SpatialFieldEnv]:
    ss = np.random.SeedSequence(seed)
    param_seeds = ss.spawn(n)
    outcome_seeds = ss.spawn(n)
    envs = []
    for i in range(n):
        prng = np.random.default_rng(param_seeds[i])
        envs.append(generate_instance(prng, seed=outcome_seeds[i],
                                      instance_id=i, **kwargs))
    return envs
