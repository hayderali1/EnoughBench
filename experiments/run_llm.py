"""Run LLM-agent baselines and write results + trajectories.

Examples:
  export OPENAI_API_KEY=sk-...
  python -m experiments.run_llm --track A --n 30 --model gpt-4o-mini
  python -m experiments.run_llm --track B --n 20 --agents hybrid

Outputs (share both with whoever writes the paper):
  results/llm_<track>_results.csv     # the metrics table row(s)
  results/llm_<track>_trajectories.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from enoughbench.metrics.metrics import summarize, accuracy, avg_cost
from enoughbench.metrics.faithfulness import faithfulness_score
from enoughbench.agents.llm_agent import (
    OpenAIClient, react_agent, uncertainty_gated_hybrid,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_track(track: str, n: int, seed: int):
    if track.upper() == "A":
        from enoughbench.tracks.hypothesis import generate_suite as gs
        from enoughbench.oracles.frontier import reference_frontier
        kw = dict(k=4, m=6, budget=12.0)
        sf = lambda: gs(seed, n=n, **kw)
        frontier = reference_frontier(lambda: gs(seed, n=max(n, 200), **kw))
    else:
        from enoughbench.tracks.spatial_field import generate_suite as gs
        from experiments.run_track_b import reference_frontier as rf_b
        kw = dict(grid=12, n_nodes=14, budget=18.0, ell=4.0, noise=0.03,
                  tol=2.0, topk=15, n_samples=200)
        sf = lambda: gs(seed, n=n, **kw)
        # reuse Track B frontier builder (uses its own seeded suite)
        frontier = rf_b()
    return sf, frontier


def traj_to_dict(t):
    d = asdict(t)
    # StepRecords already dataclasses -> asdict handles them; trim posteriors for size
    for s in d.get("steps", []):
        if s.get("posterior") and len(s["posterior"]) > 12:
            s["posterior"] = None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="A", choices=["A", "B", "a", "b"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--agents", default="react,hybrid",
                    help="comma list: react,hybrid (react is Track A only)")
    ap.add_argument("--max-steps", type=int, default=12)
    args = ap.parse_args()
    track = args.track.upper()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    client = OpenAIClient(model=args.model)
    sf, frontier = build_track(track, args.n, args.seed)
    wanted = [a.strip() for a in args.agents.split(",") if a.strip()]

    agents = {}
    if "react" in wanted and track == "A":
        agents["ReAct (LLM)"] = react_agent(client, max_steps=args.max_steps)
    if "hybrid" in wanted:
        agents["uncertainty-gated"] = uncertainty_gated_hybrid(
            client, max_steps=max(args.max_steps, 20))

    rows, all_traj = [], {}
    for name, agent in agents.items():
        trajs = [agent(env) for env in sf()]
        row = summarize(name, trajs, frontier)
        fs = faithfulness_score(trajs)
        row["faithfulness"] = round(fs, 4) if fs is not None else None
        rows.append(row)
        all_traj[name] = [traj_to_dict(t) for t in trajs]
        print(f"{name:<18} acc={row['accuracy']:.3f} cost={row['avg_cost']:.2f} "
              f"ECE={row['ece']:.3f} regret={row['sufficiency_regret']:.3f} "
              f"faith={row['faithfulness']}")

    csv_path = os.path.join(RESULTS_DIR, f"llm_{track}_results.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        fields = ["agent", "accuracy", "avg_cost", "ece",
                  "sufficiency_regret", "faithfulness"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    traj_path = os.path.join(RESULTS_DIR, f"llm_{track}_trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(all_traj, f, indent=1, default=str)

    print(f"\nAPI calls: {client.n_calls}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {traj_path}")
    print("Share both files to complete the paper's LLM table.")


if __name__ == "__main__":
    main()
