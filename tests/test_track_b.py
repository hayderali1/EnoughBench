"""Track B sanity tests."""
import numpy as np

from enoughbench.tracks.spatial_field import generate_suite
from enoughbench.agents.spatial_baselines import (
    fixed_n_agent, max_budget_agent,
)
from enoughbench.metrics.metrics import accuracy, avg_cost

KW = dict(grid=12, n_nodes=14, budget=18.0, ell=4.0, noise=0.03,
          tol=2.0, topk=15, n_samples=200)


def test_reproducible():
    a = [max_budget_agent(e) for e in generate_suite(11, n=20, **KW)]
    b = [max_budget_agent(e) for e in generate_suite(11, n=20, **KW)]
    assert accuracy(a) == accuracy(b)
    assert avg_cost(a) == avg_cost(b)


def test_budget_helps_localization():
    few = accuracy([fixed_n_agent(2)(e) for e in generate_suite(11, n=40, **KW)])
    many = accuracy([fixed_n_agent(10)(e) for e in generate_suite(11, n=40, **KW)])
    assert many >= few


def test_belief_rng_isolation():
    """Calling localize() many times must not change measurement outcomes:
    accuracy with extra belief calls equals accuracy without."""
    suite = lambda: generate_suite(11, n=15, **KW)
    base = accuracy([fixed_n_agent(5)(e) for e in suite()])

    def agent_with_extra_belief(env):
        for _ in range(5):
            env.localize()  # extra belief sampling should not perturb anything
            aff = [a for a in range(env.A)
                   if env.action_costs[a] <= env.budget_left() + 1e-9]
            if not aff:
                break
            env.localize()
            scores = env.ucb_scores()
            env.query(int(np.array(aff)[np.argmax(scores[np.array(aff)])]))
        best, conf, _ = env.localize()
        return env.report(int(best), float(conf))

    perturbed = accuracy([agent_with_extra_belief(e) for e in suite()])
    assert abs(base - perturbed) < 1e-9


if __name__ == "__main__":
    test_reproducible()
    test_budget_helps_localization()
    test_belief_rng_isolation()
    print("track B sanity tests passed")
