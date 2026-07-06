"""Track C go/no-go experiment (change-detection).

Run:  python -m experiments.run_track_c
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enoughbench.tracks.change_detection import generate_suite
from enoughbench.agents.change_baselines import (
    max_budget_agent, fixed_n_agent, threshold_agent, random_agent,
)
from enoughbench.oracles.frontier import pareto_front
from enoughbench.metrics.metrics import summarize, accuracy, avg_cost

SEED = 5
N = 300
SUITE_KW = dict(T=10, delta=1.5, sigma=1.0, budget=12.0)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def suite_factory():
    return generate_suite(SEED, n=N, **SUITE_KW)


def op(agent):
    trajs = [agent(env) for env in suite_factory()]
    return avg_cost(trajs), accuracy(trajs)


def reference_frontier():
    pts = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        pts.append(op(threshold_agent(tau)))
    for n in (0, 1, 2, 3, 4, 6, 8, 10):
        pts.append(op(fixed_n_agent(n)))
    pts.append(op(max_budget_agent))
    return [(c, a, 0.0) for (c, a) in pareto_front(pts)]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frontier = reference_frontier()
    agents = {
        "threshold(0.80)": threshold_agent(0.80),
        "threshold(0.95)": threshold_agent(0.95),
        "fixed_n(3)": fixed_n_agent(3),
        "fixed_n(6)": fixed_n_agent(6),
        "max_budget": max_budget_agent,
        "random": random_agent(stop_prob=0.25, seed=3),
    }
    rows, opd = [], {}
    for name, agent in agents.items():
        trajs = [agent(env) for env in suite_factory()]
        rows.append(summarize(name, trajs, frontier))
        opd[name] = (avg_cost(trajs), accuracy(trajs))

    with open(os.path.join(RESULTS_DIR, "track_c_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    fc = np.array([c for c, a, _ in frontier]); fa = np.array([a for c, a, _ in frontier])
    o = np.argsort(fc)
    plt.figure(figsize=(7, 5))
    plt.plot(fc[o], fa[o], "-o", color="#222", lw=2, ms=4,
             label="reference frontier (Pareto)")
    for (name, (c, a)), mk in zip(opd.items(), ["s", "^", "D", "P", "X", "o"]):
        plt.scatter([c], [a], s=70, marker=mk, label=name, zorder=3)
    plt.xlabel("average cost (slots sampled)"); plt.ylabel("change-decision accuracy")
    plt.title("EnoughBench Track C: change-detection / temporal sufficiency")
    plt.grid(alpha=0.3); plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "track_c_frontier.png"), dpi=130)

    print(f"\nSeeded suite: n={N}, {SUITE_KW}, seed={SEED}")
    print("-" * 72)
    print(f"{'agent':<18}{'acc':>8}{'cost':>8}{'ECE':>8}{'suff.regret':>14}")
    print("-" * 72)
    for r in rows:
        print(f"{r['agent']:<18}{r['accuracy']:>8}{r['avg_cost']:>8}{r['ece']:>8}"
              f"{r['sufficiency_regret']:>14}")
    print(f"frontier: {[(round(c,2), round(a,3)) for c,a,_ in frontier]}")


if __name__ == "__main__":
    main()
