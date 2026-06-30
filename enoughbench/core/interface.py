"""Core environment interface for EnoughBench.

Every track implements `Environment`. An agent (policy) interacts only through
this tiny API, which keeps the benchmark model-agnostic:

    observe()                 -> Observation (state visible to the agent)
    query(action)            -> Evidence (noisy result, at a known cost)
    budget_left()            -> remaining budget
    report(answer, conf)     -> ends the episode

The harness (`run_episode`) drives any agent against any environment and
returns a fully logged `Trajectory` for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Observation:
    """What the agent sees at a decision point."""
    question: str
    actions: List[int]            # ids of available evidence-gathering actions
    action_costs: Dict[int, float]
    budget_left: float
    posterior: Optional[List[float]] = None  # exposed for classical baselines/oracle
    step: int = 0


@dataclass
class Evidence:
    """Result of a single query."""
    action: int
    outcome: Any
    cost: float


@dataclass
class StepRecord:
    step: int
    kind: str                     # "query" or "report"
    action: Optional[int]
    outcome: Any
    cost: float
    budget_left: float
    posterior: Optional[List[float]] = None


@dataclass
class Trajectory:
    instance_id: int
    steps: List[StepRecord] = field(default_factory=list)
    answer: Optional[int] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None        # for the explanation-faithfulness metric
    total_cost: float = 0.0
    correct: Optional[bool] = None
    ground_truth: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Environment:
    """Base class for a single benchmark instance."""

    def reset(self) -> Observation:
        raise NotImplementedError

    def observe(self) -> Observation:
        raise NotImplementedError

    def query(self, action: int) -> Evidence:
        raise NotImplementedError

    def budget_left(self) -> float:
        raise NotImplementedError

    def report(self, answer: int, confidence: float, rationale: str = "") -> Trajectory:
        raise NotImplementedError


# An agent is just a callable: given an Environment, drive it and return the
# finished Trajectory. This makes LLM agents and classical policies interchangeable.
Agent = Callable[[Environment], Trajectory]


def run_episode(env: Environment, agent: Agent) -> Trajectory:
    """Run one agent on one environment instance. The agent is responsible for
    calling env.query / env.report; this wrapper exists for symmetry and future
    instrumentation (timing, API-call counting, etc.)."""
    env.reset()
    return agent(env)
