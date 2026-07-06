"""Tests for Track C (change-detection) and Track D (motion-cost robotics)."""
import numpy as np

from enoughbench.tracks.change_detection import generate_suite as gsC
from enoughbench.tracks.spatial_motion import generate_suite as gsD
from enoughbench.agents.change_baselines import (
    fixed_n_agent as cd_fixed, max_budget_agent as cd_max,
)
from enoughbench.agents.spatial_baselines import (
    fixed_n_agent as sp_fixed, max_budget_agent as sp_max,
)
from enoughbench.agents.llm_agent import MockClient, uncertainty_gated_hybrid
from enoughbench.metrics.metrics import accuracy, avg_cost

D_KW = dict(grid=12, n_nodes=14, budget=30.0, ell=4.0, noise=0.03,
            tol=2.0, topk=15, n_samples=200, travel_rate=0.7)


# ---- Track C ----
def test_c_reproducible():
    a = [cd_max(e) for e in gsC(5, n=30)]
    b = [cd_max(e) for e in gsC(5, n=30)]
    assert accuracy(a) == accuracy(b) and avg_cost(a) == avg_cost(b)


def test_c_budget_helps():
    few = accuracy([cd_fixed(1)(e) for e in gsC(5, n=80)])
    many = accuracy([cd_fixed(8)(e) for e in gsC(5, n=80)])
    assert many >= few


def test_c_posterior_normalized():
    env = gsC(5, n=1)[0]
    env.reset()
    p = env.posterior()
    assert abs(p.sum() - 1.0) < 1e-9
    assert 0.0 <= env.p_change() <= 1.0


def test_c_hybrid_mock():
    script = [{"action": "continue", "rationale": "uncertain"},
              {"action": "stop", "rationale": "sufficient evidence"}]
    env = gsC(5, n=1)[0]
    t = uncertainty_gated_hybrid(MockClient(script))(env)
    assert t.answer in (0, 1)
    assert 0.0 <= t.confidence <= 1.0


# ---- Track D ----
def test_d_reproducible():
    a = [sp_max(e) for e in gsD(13, n=12, **D_KW)]
    b = [sp_max(e) for e in gsD(13, n=12, **D_KW)]
    assert accuracy(a) == accuracy(b) and avg_cost(a) == avg_cost(b)


def test_d_travel_cost_changes_with_position():
    """A robot waypoint's cost must depend on the robot's current position."""
    env = gsD(13, n=1, **D_KW)[0]
    env.reset()
    c_before = env.action_costs[env.n_nodes:].copy()   # robot-waypoint costs
    env.query(env.n_nodes)                              # move robot to a waypoint
    c_after = env.action_costs[env.n_nodes:]
    assert not np.allclose(c_before, c_after)           # costs shifted after moving


def test_d_hybrid_mock():
    script = [{"action": "continue", "rationale": "uncertain"},
              {"action": "stop", "rationale": "sufficient"}]
    env = gsD(13, n=1, **D_KW)[0]
    t = uncertainty_gated_hybrid(MockClient(script))(env)
    assert t.answer is not None and t.total_cost > 0


if __name__ == "__main__":
    for fn in [test_c_reproducible, test_c_budget_helps, test_c_posterior_normalized,
               test_c_hybrid_mock, test_d_reproducible,
               test_d_travel_cost_changes_with_position, test_d_hybrid_mock]:
        fn()
    print("Track C/D tests passed")
