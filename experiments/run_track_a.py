"""Days 1-4 go/no-go experiment for Track A.

Builds the oracle accuracy-vs-budget frontier, runs the non-LLM baselines on an
identical seeded suite, computes the metric suite (incl. sufficiency regret),
writes results/track_a_results.csv and results/track_a_frontier.png.

Run:  python -m experiments.run_track_a
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enoughbench.tracks.hypothesis import generate_suite
from enoughbench.oracles.bayes_sequential import bayes_risk_oracle
from enoughbench.oracles.frontier import reference_frontier
from enoughbench.agents.baselines import (
    max_budget_agent, fixed_n_agent, threshold_agent, random_agent,
)
from enoughbench.metrics.metrics import summarize, accuracy, avg_cost

SEED = 7
N = 300
SUITE_KW = dict(k=4, m=6, budget=12.0)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def suite_factory():
    return generate_suite(SEED, n=N, **SUITE_KW)


def run_agent(name, agent, frontier):
    trajs = [agent(env) for env in suite_factory()]
    return summarize(name, trajs, frontier), trajs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Reference frontier = Pareto front over strong model-aware policies.
    frontier = reference_frontier(suite_factory)

    # 2) Baselines (all share info-greedy selection; differ only in stopping).
    agents = {
        "oracle(lam=8)": bayes_risk_oracle(8.0),
        "oracle(lam=20)": bayes_risk_oracle(20.0),
        "threshold(0.80)": threshold_agent(0.80),
        "threshold(0.95)": threshold_agent(0.95),
        "fixed_n(2)": fixed_n_agent(2),
        "fixed_n(5)": fixed_n_agent(5),
        "max_budget": max_budget_agent,
        "random": random_agent(stop_prob=0.25, seed=1),
    }

    rows = []
    operating_points = {}
    for name, agent in agents.items():
        row, trajs = run_agent(name, agent, frontier)
        rows.append(row)
        operating_points[name] = (avg_cost(trajs), accuracy(trajs))

    # 3) Write CSV.
    csv_path = os.path.join(RESULTS_DIR, "track_a_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4) Plot the frontier + baseline operating points.
    fcost = [c for c, a, _ in frontier]
    facc = [a for c, a, _ in frontier]
    order = np.argsort(fcost)
    fcost = np.array(fcost)[order]
    facc = np.array(facc)[order]

    plt.figure(figsize=(7, 5))
    plt.plot(fcost, facc, "-o", color="#222", lw=2, ms=4,
             label="reference frontier (Pareto, model-aware policies)")
    markers = ["s", "^", "v", "D", "P", "X", "*", "o"]
    for (name, (c, a)), mk in zip(operating_points.items(), markers):
        plt.scatter([c], [a], s=70, marker=mk, label=name, zorder=3)
    plt.xlabel("average cost (tests bought)")
    plt.ylabel("accuracy")
    plt.title("EnoughBench Track A: accuracy vs. budget frontier")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "track_a_frontier.png")
    plt.savefig(png_path, dpi=130)

    # 5) Console summary.
    print(f"\nSeeded suite: n={N}, {SUITE_KW}, seed={SEED}")
    print("-" * 72)
    hdr = f"{'agent':<18}{'acc':>8}{'cost':>8}{'ECE':>8}{'suff.regret':>14}"
    print(hdr); print("-" * 72)
    for r in rows:
        print(f"{r['agent']:<18}{r['accuracy']:>8}{r['avg_cost']:>8}"
              f"{r['ece']:>8}{r['sufficiency_regret']:>14}")
    print("-" * 72)
    print(f"frontier points: {[(round(c,2), round(a,3)) for c,a,_ in frontier]}")
    print(f"\nwrote: {csv_path}")
    print(f"wrote: {png_path}")


if __name__ == "__main__":
    main()
