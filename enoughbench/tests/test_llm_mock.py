"""Validate the LLM agent control flow with a MockClient (no network).

Checks parsing, action dispatch, budget handling, stopping, fallback, and
integration with both tracks' environments and the metric suite.
"""
import numpy as np

from enoughbench.tracks.hypothesis import generate_suite as gsA
from enoughbench.tracks.spatial_field import generate_suite as gsB
from enoughbench.agents.llm_agent import (
    MockClient, react_agent, uncertainty_gated_hybrid,
)
from enoughbench.metrics.metrics import accuracy
from enoughbench.metrics.faithfulness import faithfulness_score


def test_react_mock_track_a():
    # Script: two queries then a report.
    script = [
        {"action": "query", "target": 0},
        {"action": "query", "target": 1},
        {"action": "report", "answer": 2, "confidence": 0.7, "rationale": "sufficient evidence"},
    ]
    env = gsA(7, n=1, k=4, m=6, budget=12.0)[0]
    t = react_agent(MockClient(script))(env)
    assert t.answer == 2
    assert t.confidence == 0.7
    assert t.total_cost == 2.0          # two queries at cost 1
    assert t.rationale == "sufficient evidence"


def test_react_mock_fallback_terminates():
    # Empty script -> client always returns terminal 'report'; must terminate.
    env = gsA(7, n=1, k=4, m=6, budget=12.0)[0]
    t = react_agent(MockClient(script=[]))(env)
    assert t.answer is not None
    assert t.confidence is not None


def test_hybrid_mock_track_a():
    # Continue twice, then stop. Classical posterior supplies the answer.
    script = [
        {"action": "continue", "rationale": "still uncertain"},
        {"action": "continue", "rationale": "still uncertain"},
        {"action": "stop", "rationale": "confident enough"},
    ]
    env = gsA(7, n=1, k=4, m=6, budget=12.0)[0]
    t = uncertainty_gated_hybrid(MockClient(script))(env)
    assert t.total_cost == 2.0          # two 'continue' -> two queries
    assert 0.0 <= t.confidence <= 1.0
    assert t.rationale == "confident enough"


def test_hybrid_mock_track_b():
    script = [
        {"action": "continue", "rationale": "uncertain"},
        {"action": "continue", "rationale": "uncertain"},
        {"action": "stop", "rationale": "sufficient"},
    ]
    env = gsB(11, n=1, grid=12, n_nodes=14, budget=18.0, ell=4.0,
              noise=0.03, tol=2.0, topk=15, n_samples=200)[0]
    t = uncertainty_gated_hybrid(MockClient(script))(env)
    assert t.answer is not None
    assert t.total_cost > 0
    assert 0.0 <= t.confidence <= 1.0


def test_faithfulness_runs_on_llm_trajectories():
    script = [{"action": "report", "answer": 0, "confidence": 0.9,
               "rationale": "sufficient evidence, confident"}]
    env = gsA(7, n=1, k=4, m=6, budget=12.0)[0]
    t = react_agent(MockClient(script))(env)
    fs = faithfulness_score([t])
    assert fs is None or (0.0 <= fs <= 1.0)


if __name__ == "__main__":
    test_react_mock_track_a()
    test_react_mock_fallback_terminates()
    test_hybrid_mock_track_a()
    test_hybrid_mock_track_b()
    test_faithfulness_runs_on_llm_trajectories()
    print("LLM mock tests passed")
