# EnoughBench

A reproducible, **hardware-free** benchmark for **budget-aware stopping,
confidence calibration, and explanation faithfulness in LLM agents**. It scores
not just whether an agent is *right*, but *when it decides it has gathered enough
evidence to commit* — measured as **sufficiency regret** against a best-known
reference frontier — and whether its stated confidence and stated reason are
trustworthy.

**Status:** Four tracks (A–D) implemented with classical baselines, a full
metric suite, an OpenAI LLM-agent layer, and 19 passing tests. Runs on commodity
hardware; no robots, sensors, or GUI simulators required.

---

## Contents
1. [Install](#install)
2. [Quick start (60 seconds)](#quick-start-60-seconds)
3. [Full results pipeline](#full-results-pipeline)
4. [What you get](#what-you-get)
5. [Benchmark design](#benchmark-design)
6. [Metrics](#metrics)
7. [Reproducibility](#reproducibility)
8. [Troubleshooting](#troubleshooting)
9. [Repo layout](#repo-layout)
10. [Citation](#citation)

---

## Install

Python 3.9+.

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e .          # installs numpy, scipy, matplotlib
pip install openai        # only needed for the LLM baselines
```

Verify everything works (no network or API key needed):

```bash
pytest -q                 # 12 tests: Track A, Track B, and the LLM loop (mocked)
```

---

## Quick start (60 seconds)

Reproduce both classical frontiers — no API key required:

```bash
python -m experiments.run_track_a    # -> results/track_a_frontier.png + track_a_results.csv
python -m experiments.run_track_b    # -> results/track_b_frontier.png + track_b_results.csv
python -m experiments.run_track_c    # -> results/track_c_frontier.png + track_c_results.csv
python -m experiments.run_track_d    # -> results/track_d_frontier.png + track_d_results.csv
```

Each prints a metrics table and writes a frontier plot. This validates the
benchmark instrument end-to-end.

---

## Full results pipeline

The classical tracks above run anywhere. The **LLM baselines call the OpenAI
API**, so they run on your machine with your key.

### Step 1 — set your key

```bash
export OPENAI_API_KEY=sk-...            # macOS/Linux
# setx OPENAI_API_KEY "sk-..."          # Windows (new shell afterwards)
```

### Step 2 — run the LLM agents

```bash
python -m experiments.run_llm --track A --n 150 --model gpt-4o-mini
python -m experiments.run_llm --track B --n 100 --agents hybrid --model gpt-4o-mini
python -m experiments.run_llm --track C --n 150 --agents hybrid --model gpt-4o-mini
python -m experiments.run_llm --track D --n 100 --agents hybrid --model gpt-4o-mini
```

Options:

| flag | default | meaning |
|------|---------|---------|
| `--track` | `A` | `A` hypothesis testing, `B` spatial/IoT, `C` change-detection, `D` motion-cost robotics |
| `--n` | `30` | number of instances (n ≥ 100 keeps ECE/regret stable) |
| `--model` | `gpt-4o-mini` | any OpenAI chat model |
| `--agents` | `react,hybrid` | comma list; `react` is Track A only |
| `--max-steps` | `12` | per-episode action cap (cost control) |
| `--seed` | `7` | suite seed |

**Agents.**
- `react` — the LLM is given the likelihood model and the evidence history, and
  it chooses both *which* action to take and *when* to stop, then reports an
  answer + calibrated confidence. (Track A only; discrete answer space.)
- `hybrid` — a classical inner loop selects measurements and computes the
  posterior/answer; the LLM only **gates** stop/continue and emits a rationale.
  (All four tracks.)

**Cost.** On `gpt-4o-mini`, well under a couple of dollars for all four tracks.
The runs are network-bound (one API call at a time), so the full set takes
roughly 20–40 minutes; lower `--n` to go faster.

### Step 3 — share the outputs

Send back the two files per track (below). They contain everything needed to fill
in the paper's LLM results table and the explanation-faithfulness column.

---

## What you get

Running `run_llm` writes, per track:

```
results/llm_A_results.csv          # one row per agent: accuracy, cost, ECE, regret, faithfulness
results/llm_A_trajectories.json    # full per-episode logs incl. stop-rationales
```

`run_track_a` / `run_track_b` write the classical tables + frontier PNGs. A
results row looks like:

```
agent,accuracy,avg_cost,ece,sufficiency_regret,faithfulness
ReAct (LLM),0.71,4.30,0.14,0.86,0.78
uncertainty-gated,0.79,3.10,0.06,0.20,0.91
```

(numbers illustrative — yours come from your run.)

---

## Benchmark design

### Interface (what an agent implements)

```
observe()                                 -> question, available actions, costs, budget
query(action)                             -> noisy evidence at a known cost
budget_left()                             -> remaining budget
report(answer, confidence, rationale="")  -> ends the episode
```

An agent is any callable `env -> Trajectory`, so classical policies and LLM
agents are fully interchangeable.

### Track A — sequential hypothesis testing
`k` hypotheses, one latent truth; `m` repeatable noisy Bernoulli tests with
costs. The agent buys tests, maintains an exact Bayesian posterior, and decides
when to stop and report. Reskins to a concrete diagnostic story with no change to
the math.

### Track B — spatial-field localization via simulated IoT coordination
A Gaussian-random-field over a grid is the latent quantity. A network of fixed
IoT nodes (cheap) plus robot-reachable waypoints (costlier) are candidate
measurement locations under an energy budget. The agent coordinates the
network — choosing where to measure — maintains a Gaussian-process posterior, and
reports the cell containing the field maximum (within a tolerance). All
simulated; no ROS/Gazebo/hardware.

### Track C — change-detection / temporal sufficiency
A scalar signal is observed over discrete time slots; with probability 0.5 its
mean jumps at an unknown change-time. The agent actively samples slots and must
decide when it has enough evidence to declare "changed" or "no change", with a
calibrated confidence. Adds a *temporal* sufficiency decision (online-monitoring
/ change-point detection) absent from A and B; exact Bayesian posterior over
change-points.

### Track D — motion-cost robotics (path-dependent spatial sensing)
Same GP field as Track B, but embodied: fixed IoT nodes transmit cheaply, while a
mobile robot pays a travel cost proportional to the distance from its current
position to the next measurement, then moves there. The agent must couple *where
to measure* with *where it already is* (informative path planning) on top of the
stopping decision. Reuses Track B's belief; only the cost model differs.

---

## Metrics

- **accuracy / avg_cost** — task quality and effort.
- **sufficiency regret** *(headline)* — excess cost vs the reference frontier at
  equal accuracy: `max(0, agent_cost - frontier_cost_at(agent_accuracy))`.
  Zero on the frontier; grows with wasted budget.
- **ECE** — expected calibration error of the reported confidence at stop time.
- **explanation faithfulness** — is the agent's stated stop-rationale consistent
  with its actual evidence state? (applies to rationale-emitting / LLM agents.)

**Reference frontier.** The Pareto front over a basket of strong **model-aware**
policies (confidence-threshold, fixed-`n`, and a Bayes-risk oracle for Track A).
Near-optimal policies sit on it (regret ≈ 0); the gap opens for weak policies and
for LLM agents reasoning through the API without the true model.

---

## Reproducibility

Every instance has independent, deterministic parameter and outcome seeds, so
results do **not** depend on agent behavior or instance order. Belief sampling
(Track B) uses a separate RNG from measurement noise. Experiment scripts fix the
suite seed. The frozen suites and one-command runners reproduce every table and
figure.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: openai` | `pip install openai` |
| `OpenAIError: api_key` | `export OPENAI_API_KEY=sk-...` in the same shell |
| `model not found` | check `--model` (e.g. `gpt-4o-mini`, `gpt-4o`) |
| `json.JSONDecodeError` during a run | rare; the loop retries and falls back. If frequent, send the traceback — it's a one-line prompt/parse tweak |
| Plot errors on a headless server | already handled (`matplotlib Agg` backend); PNGs are written, not shown |
| Want lower cost | reduce `--n` and `--max-steps` |

The non-LLM code is fully tested. The only path that can't be tested without a
key is the live API call itself; if the first LLM run errors, send the traceback.

---

## Repo layout

```
enoughbench/
  core/interface.py            # Observation/Evidence/Trajectory, Environment, run_episode
  tracks/hypothesis.py         # Track A environment + seeded generators
  tracks/spatial_field.py      # Track B: GP spatial field + IoT coordination
  tracks/change_detection.py   # Track C: change-detection / temporal sufficiency
  tracks/spatial_motion.py     # Track D: motion-cost robotics (path-dependent)
  agents/selection.py          # info-gain-per-cost selection (Track A)
  agents/baselines.py          # Track A baselines: max_budget, fixed_n, threshold, random
  agents/spatial_baselines.py  # Track B/D baselines (GP-UCB selection)
  agents/change_baselines.py   # Track C baselines
  agents/llm_agent.py          # ReAct + uncertainty-gated hybrid (OpenAI) + MockClient
  oracles/bayes_sequential.py  # one-step Bayes-risk oracle (Track A)
  oracles/frontier.py          # Pareto reference frontier
  metrics/metrics.py           # accuracy, cost, ECE, sufficiency regret
  metrics/faithfulness.py      # explanation faithfulness (XAI dimension)
experiments/run_track_{a,b,c,d}.py   # per-track frontier experiments
experiments/run_llm.py         # LLM baselines (your OpenAI key), tracks A-D
tests/                         # 19 tests incl. mocked LLM loop on all tracks
```

---

## Citation

```bibtex
@misc{enoughbench2026,
  title  = {EnoughBench: A Benchmark for Evaluating When LLM Agents Should Stop Gathering Evidence},
  author = {Hayder Almamori},
  year   = {2026},
  note   = {Code repository accompanying the paper. Submitted to IDAP 2026.},
  howpublished = {\url{https://github.com/hayderali1/EnoughBench}}
}
```

## License

MIT — see `LICENSE`.
