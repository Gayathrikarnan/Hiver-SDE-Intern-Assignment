import sys
from pathlib import Path

# Allow Python to import files from the scripts folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import agent


def test_agent_returns_required_fields():
    result = agent.run_agent(
        "I paid for Spotify Premium but my account is still showing the free plan."
    )

    required_fields = [
        "customer_message",
        "intent",
        "confidence",
        "similarity",
        "reply",
        "decision",
        "reason",
        "source_case",
    ]

    for field in required_fields:
        assert field in result


def test_confidence_is_valid():
    result = agent.run_agent(
        "Spotify keeps crashing whenever I open the app."
    )

    assert 0 <= result["confidence"] <= 1


def test_similarity_is_valid():
    result = agent.run_agent(
        "Can you add lyrics translation to Spotify?"
    )

    assert 0 <= result["similarity"] <= 1


def test_reply_is_not_empty():
    result = agent.run_agent(
        "Where can I find my playlists?"
    )

    assert isinstance(result["reply"], str)
    assert len(result["reply"].strip()) > 0


def test_decision_is_valid():
    result = agent.run_agent(
        "I paid for Spotify Premium but my account is still showing the free plan."
    )

    assert result["decision"] in ["AUTO-HANDLE", "ESCALATE"]