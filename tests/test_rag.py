"""Five easy questions asked to the real RAG. Needs Ollama running: if it isn't, the tests fail with a clear message."""
from pathlib import Path

import numpy as np
import pytest

import rag
from tests.cases import CASES, Case, topic_of


@pytest.fixture(scope="module")
def index() -> tuple[list[str], np.ndarray]:
    """Function that embeds the docs once and shares them with every test (so it's faster).
    If Ollama is down, this raises OllamaError and every test fails saying so (instead of being skipped).
    """
    chunks = rag.load_chunks(Path(__file__).parent.parent / "docs")
    return chunks, rag.embed(chunks)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["question"])
def test_easy_question(case: Case, index: tuple[list[str], np.ndarray]) -> None:
    """Checks that the right chunk is retrieved and that the answer contains the expected text."""
    chunks, vectors = index
    hits = rag.retrieve(case["question"], chunks, vectors, rag.TOP_K)
    if case["expected_topic"]:
        found = [topic_of(chunks[i]) for i, _ in hits]
        assert case["expected_topic"] in found, f"retrieved {found}"
    answer = rag.ask_llm(rag.build_prompt(case["question"], [chunks[i] for i, _ in hits])).lower()
    assert any(s in answer for s in case["answer_contains"]), answer
