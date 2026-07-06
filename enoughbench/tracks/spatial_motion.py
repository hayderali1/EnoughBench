"""Track D -- motion-cost robotics (path-dependent spatial sensing).

Same GP spatial-field localization as Track B, but the cost model is embodied:
fixed IoT nodes transmit cheaply (no robot motion), while a mobile robot pays a
travel cost proportional to the distance from its CURRENT position to the next
measurement, then moves there. The agent must therefore couple *where to measure*
with *where it already is* -- the informative-path-planning (IPP) decision -- on
top of the sufficiency (when-to-stop) decision.

This reuses Track B's GP belief and tolerance-aware localization unchanged; only
the cost model and (dynamic) selection differ. No ROS/Gazebo/hardware.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .spatial_field import SpatialFieldEnv, _rbf
from ..core.interface import Evidence, StepRecord, Trajectory


class MotionSpatialEnv(SpatialFieldEnv):
    def __init__(self, *, n_nodes: int, travel_rate: float, node_cost: float,
                 robot_base: float, grid: int, **kw):
        self.n_nodes = n_nodes
        self.travel_rate = travel_rate
        self.node_cost = node_cost
        self.robot_base = robot_base
        self.start_pos = np.array([grid // 2, grid // 2], dtype=float)
        # initial (pre-travel) cost array; refreshed each step from robot pos
        super().__init__(grid=grid, **kw)

    def _reset_state(self):
        super()._reset_state()
        self.robot_pos = self.start_pos.copy()
        self._refresh_costs()

    def _refresh_costs(self):
        costs = np.empty(self.A)
        costs[:self.n_nodes] = self.node_cost
        for a in range(self.n_nodes, self.A):
            d = float(np.hypot(*(self.action_locs[a] - self.robot_pos)))
            costs[a] = self.robot_base + self.travel_rate * d
        self.action_costs = costs
        self.costs = costs

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
        if a >= self.n_nodes:                 # robot action -> robot travels there
            self.robot_pos = loc.astype(float).copy()
        self._refresh_costs()                 # costs now relative to new position
        self._traj.steps.append(
            StepRecord(self._step, "query", a, round(val, 4), cost,
                       self.budget - self._spent, None))
        return Evidence(action=a, outcome=val, cost=cost)


def generate_instance(param_rng, seed, grid=12, n_nodes=12, budget=30.0,
                      sig2=1.0, ell=4.0, noise=0.03, node_cost=1.0,
                      robot_base=1.0, travel_rate=0.7, instance_id=0,
                      tol=2.0, topk=15, n_samples=200):
    cells = np.array([(i, j) for i in range(grid) for j in range(grid)], dtype=float)
    K = _rbf(cells, cells, sig2, ell) + 1e-6 * np.eye(len(cells))
    L = np.linalg.cholesky(K)
    f_true = L @ param_rng.standard_normal(len(cells))
    node_idx = param_rng.choice(len(cells), size=n_nodes, replace=False)
    node_locs = cells[node_idx]
    step = max(2, grid // 4)
    wp = np.array([(i, j) for i in range(0, grid, step) for j in range(0, grid, step)],
                  dtype=float)
    action_locs = np.vstack([node_locs, wp])
    base_costs = np.concatenate([np.full(len(node_locs), node_cost),
                                 np.full(len(wp), robot_base)])
    return MotionSpatialEnv(
        grid=grid, action_locs=action_locs, action_costs=base_costs,
        f_true=f_true, budget=budget, sig2=sig2, ell=ell, noise=noise,
        seed=seed, instance_id=instance_id, tol=tol, topk=topk, n_samples=n_samples,
        n_nodes=len(node_locs), travel_rate=travel_rate, node_cost=node_cost,
        robot_base=robot_base,
    )


def generate_suite(seed: int, n: int = 80, **kw) -> List[MotionSpatialEnv]:
    ss = np.random.SeedSequence(seed)
    pseeds, oseeds = ss.spawn(n), ss.spawn(n)
    return [generate_instance(np.random.default_rng(pseeds[i]), seed=oseeds[i],
                              instance_id=i, **kw) for i in range(n)]
