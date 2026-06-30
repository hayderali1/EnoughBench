"""LLM agent baselines for EnoughBench (OpenAI / gpt-4o-mini by default).

Two agents, both interacting through the standard observe/query/report API:

  react_agent
      The LLM is given the likelihood model and the evidence history, and it
      chooses BOTH which action to take and WHEN to stop, then reports an answer
      and a calibrated confidence. This is the headline "does the LLM know when
      it has enough?" baseline (Track A; the discrete answer space makes it
      well-posed for a language model).

  uncertainty_gated_hybrid
      A classical inner loop selects measurements and computes the posterior /
      belief (exact Bayes in Track A, GP in Track B); the LLM is given the
      current confidence + budget and only GATES the stop decision, emitting a
      rationale. This is the proposed reference hybrid and works on both tracks.

This module cannot be exercised against the real API inside the build sandbox
(no network to api.openai.com); it is validated with `MockClient` in the tests,
which checks every part of the loop except the model call itself.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.interface import Environment, Trajectory


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Thin wrapper around the OpenAI Chat Completions API.

    Reads OPENAI_API_KEY from the environment. Requests JSON-object responses and
    parses them, with simple retries. Tracks the number of API calls so cost can
    be reported.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 max_tokens: int = 250, retries: int = 2):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for LLM baselines. "
                "Install with: pip install openai"
            ) from e
        self._client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.n_calls = 0

    def decide(self, system: str, user: str) -> Dict[str, Any]:
        self.n_calls += 1
        last_err: Optional[Exception] = None
        for _ in range(self.retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return _parse_json(resp.choices[0].message.content)
            except Exception as e:  # pragma: no cover (network)
                last_err = e
        raise RuntimeError(f"OpenAI call failed: {last_err}")


class MockClient:
    """Scripted client for tests. `script` is a list of decision dicts; once
    exhausted it returns a 'report'/'stop' decision so loops always terminate."""

    def __init__(self, script: Optional[List[Dict[str, Any]]] = None,
                 terminal: Optional[Dict[str, Any]] = None):
        self.script = list(script or [])
        self.terminal = terminal or {"action": "report", "answer": 0,
                                     "confidence": 0.5, "rationale": "done"}
        self.n_calls = 0

    def decide(self, system: str, user: str) -> Dict[str, Any]:
        self.n_calls += 1
        if self.script:
            return self.script.pop(0)
        return dict(self.terminal)


def _parse_json(text: str) -> Dict[str, Any]:
    """Robust-ish JSON extraction: strip code fences, take the first {...}."""
    if text is None:
        raise ValueError("empty model response")
    t = text.strip().strip("`")
    if t.startswith("json"):
        t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start:end + 1]
    return json.loads(t)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_spatial(env: Environment) -> bool:
    return hasattr(env, "localize")


def _affordable(env: Environment):
    bl = env.budget_left()
    return [a for a in range(env.m) if env.costs[a] <= bl + 1e-9]


def _clip01(x) -> float:
    try:
        return float(min(1.0, max(0.0, float(x))))
    except (TypeError, ValueError):
        return 0.25


# --------------------------------------------------------------------------- #
# ReAct agent (Track A)
# --------------------------------------------------------------------------- #
_REACT_SYSTEM = (
    "You are an agent identifying which hypothesis is true while spending as "
    "little as possible. You know each test's outcome probabilities under each "
    "hypothesis. Each query costs budget. Reason about the evidence and decide "
    "to run another test or to stop and report. Respond ONLY with a JSON object: "
    'either {"action":"query","target":<test_id>} or '
    '{"action":"report","answer":<hypothesis_id>,"confidence":<0..1>,'
    '"rationale":"<short reason>"}. Make confidence reflect your true probability '
    "that the answer is correct."
)


def _react_user_prompt(env: Environment, history: List[Dict[str, Any]]) -> str:
    k, m = env.p.shape
    lines = [
        f"Hypotheses: {k}. Tests: {m}. Budget left: {env.budget_left():.1f}.",
        "Test outcome probabilities P(outcome=1 | hypothesis, test):",
    ]
    for i in range(k):
        row = ", ".join(f"t{j}:{env.p[i, j]:.2f}" for j in range(m))
        lines.append(f"  H{i}: {row}")
    lines.append("Test costs: " + ", ".join(f"t{j}:{env.costs[j]:.1f}" for j in range(m)))
    if history:
        obs = "; ".join(f"t{h['target']}->{h['outcome']}" for h in history)
        lines.append(f"Observations so far: {obs}")
    else:
        lines.append("Observations so far: none")
    lines.append("Choose the next action as JSON.")
    return "\n".join(lines)


def react_agent(client, max_steps: int = 12):
    """LLM chooses actions and stopping; reports answer + confidence (Track A)."""

    def agent(env: Environment) -> Trajectory:
        if _is_spatial(env):
            raise ValueError("react_agent targets Track A (discrete hypotheses).")
        history: List[Dict[str, Any]] = []
        fallbacks = 0
        for _ in range(max_steps):
            if not _affordable(env):
                break
            try:
                d = client.decide(_REACT_SYSTEM, _react_user_prompt(env, history))
            except Exception:
                fallbacks += 1
                break
            act = str(d.get("action", "report")).lower()
            if act == "query":
                tgt = int(d.get("target", 0)) % env.m
                if env.costs[tgt] > env.budget_left() + 1e-9:
                    aff = _affordable(env)
                    if not aff:
                        break
                    tgt = min(aff, key=lambda a: env.costs[a])
                ev = env.query(tgt)
                history.append({"target": tgt, "outcome": ev.outcome})
            else:
                ans = int(d.get("answer", 0)) % env.k
                conf = _clip01(d.get("confidence", 0.5))
                return env.report(ans, conf, str(d.get("rationale", "")))
        # Fallback report (only on exhaustion/error): use observed-majority guess.
        post = env.posterior()
        env._traj.meta["llm_fallback"] = fallbacks
        return env.report(int(np.argmax(post)), float(post.max()),
                          "fallback: stopped without explicit LLM report")

    return agent


# --------------------------------------------------------------------------- #
# Uncertainty-gated hybrid (Tracks A and B)
# --------------------------------------------------------------------------- #
_GATE_SYSTEM = (
    "You decide whether an agent has gathered enough evidence. You are given the "
    "current best-answer confidence, the cost spent, and the budget left. Stop "
    "when the marginal value of another measurement is low relative to its cost; "
    "do not over-gather, but do not stop while very uncertain. Respond ONLY with "
    'JSON: {"action":"stop"|"continue","rationale":"<short reason>"}.'
)


def _gate_user_prompt(conf: float, spent: float, budget_left: float,
                      cheapest: float) -> str:
    return (
        f"Current best-answer confidence: {conf:.2f}. Cost spent: {spent:.1f}. "
        f"Budget left: {budget_left:.1f}. Cheapest next measurement: {cheapest:.1f}. "
        "Decide as JSON."
    )


def _classical_state(env: Environment):
    """Return (best_answer, confidence, select_fn) for the active track."""
    if _is_spatial(env):
        best, conf, _ = env.localize()

        def select():
            aff = _affordable(env)
            scores = env.ucb_scores()
            aff = np.array(aff)
            return int(aff[np.argmax(scores[aff])])

        return int(best), float(conf), select
    else:
        from .selection import best_test
        post = env.posterior()
        best, conf = int(np.argmax(post)), float(post.max())

        def select():
            return best_test(env.posterior(), env.p, env.costs)

        return best, conf, select


def uncertainty_gated_hybrid(client, max_steps: int = 20):
    """Classical selection + classical answer; the LLM gates stop/continue."""

    def agent(env: Environment) -> Trajectory:
        last_rationale = ""
        for _ in range(max_steps):
            best, conf, select = _classical_state(env)
            aff = _affordable(env)
            if not aff:
                break
            cheapest = min(env.costs[a] for a in aff)
            try:
                d = client.decide(
                    _GATE_SYSTEM,
                    _gate_user_prompt(conf, env.budget - env.budget_left()
                                      if hasattr(env, "budget") else 0.0,
                                      env.budget_left(), cheapest),
                )
                decision = str(d.get("action", "continue")).lower()
                last_rationale = str(d.get("rationale", ""))
            except Exception:
                decision = "stop"
                last_rationale = "fallback: gate error"
            if decision == "stop":
                break
            env.query(select())
        best, conf, _ = _classical_state(env)
        return env.report(int(best), float(conf), last_rationale)

    return agent
