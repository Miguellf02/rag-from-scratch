"""The questions we ask the RAG, and what we expect back. Shared by the tests and the results report."""
import re
from typing import TypedDict


class Case(TypedDict):
    question: str
    expected_topic: str | None  # the chunk that should be retrieved (None = nothing relevant)
    answer_contains: list[str]  # at least one of these should appear in the answer (lowercase)


CASES: list[Case] = [
    {
        "question": "What is the QBER threshold for BB84?",
        "expected_topic": "QBER",
        "answer_contains": ["11"],
    },
    {
        "question": "How much light is lost per kilometre in optical fibre?",
        "expected_topic": "Secret key rate vs distance",
        "answer_contains": ["0.2"],
    },
    {
        "question": "Which machine learning models can detect anomalies in QKD telemetry?",
        "expected_topic": "Monitoring & anomaly detection",
        "answer_contains": ["isolation forest", "autoencoder"],
    },
    {
        "question": "What QBER does an intercept-resend attack on every signal cause?",
        "expected_topic": "Intercept-resend attack",
        "answer_contains": ["25"],
    },
    {
        "question": "What is the capital of France?",
        "expected_topic": None,
        "answer_contains": ["don't know", "do not know"],
    },
]


def topic_of(chunk: str) -> str:
    """Function that gets the topic name from a chunk (the bold part at the start, e.g. **QBER:** gives QBER)."""
    match = re.match(r"\*\*(.+?):\*\*", chunk)
    return match.group(1) if match else "?"
