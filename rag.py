"""Minimal RAG from scratch: embed the docs, find the closest chunks, ask the LLM."""
from pathlib import Path
from typing import Any

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"  # Ollama runs on this computer and listens on this address
EMBED_MODEL = "nomic-embed-text"  # the model that turns text into numbers
CHAT_MODEL = "qwen3:4b-instruct"  # the model that writes the answer
TOP_K = 3  # how many chunks we hand to the LLM for each question


class OllamaError(RuntimeError):
    """Error for when Ollama is not running or a model is missing (the message says how to fix it)."""


def post_ollama(endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Function that sends a request to Ollama and turns its two usual failures into clear messages
    (instead of a long traceback).
    """
    try:
        r = requests.post(f"{OLLAMA_URL}/api/{endpoint}", json=payload, timeout=timeout)
    except requests.ConnectionError:
        raise OllamaError(f"Ollama is not running at {OLLAMA_URL}. Start the Ollama app and try again.") from None
    if r.status_code == 404:  # Ollama answers 404 when the model hasn't been downloaded
        raise OllamaError(f"Model '{payload['model']}' not found. Run: ollama pull {payload['model']}")
    r.raise_for_status()
    return r.json()


def embed(texts: list[str]) -> np.ndarray:
    """Function to convert a list of texts into numbers (one row per text),
    using the nomic-embed-text model from Ollama.
    """
    return np.array(post_ollama("embed", {"model": EMBED_MODEL, "input": texts}, timeout=300)["embeddings"])


def ask_llm(prompt: str) -> str:
    """Function used to send the full prompt (instructions + retrieved chunks + your question)
    to the chat model (in this case the free qwen3, running locally through Ollama)
    and return its answer as text.
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42},  # same question, same answer (so tests are repeatable)
    }
    return post_ollama("chat", payload, timeout=600)["message"]["content"]


def load_chunks(folder: Path) -> list[str]:
    """Function that reads every .md file in the folder (docs/) and splits it at every blank line.
    If a file has no blank lines, the whole file ends up as one giant chunk.
    """
    chunks: list[str] = []
    for path in sorted(folder.glob("*.md")):  # sorted so chunk numbers are the same on every run
        chunks += [c.strip() for c in path.read_text(encoding="utf-8").split("\n\n") if c.strip()]
    # Throw away chunks that are only a title: a title looks a bit like every
    # question, so it would keep showing up in the results without answering anything.
    return [c for c in chunks if not (c.startswith("#") and "\n" not in c)]


def normalize(m: np.ndarray) -> np.ndarray:
    """Function that resizes every vector to length 1, so only its direction (the meaning) counts.
    That way, multiplying two vectors gives their similarity score directly.
    """
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def retrieve(question: str, chunks: list[str], vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Function that converts your question into numbers and returns the k chunks with the best score
    (k = TOP_K = 3), as (chunk position, score) pairs. The most important function in the script.
    """
    if len(chunks) != len(vectors):  # a real check, not an assert (asserts are skipped with python -O)
        raise ValueError("chunks and vectors must stay aligned")
    # We use cosine because it compares where two vectors point, whereas Euclidean
    # distance also reacts to how long they are. With length-1 vectors both give the
    # same ranking, and cosine is just one matrix multiplication: one score per chunk.
    scores = normalize(vectors) @ normalize(embed([question]))[0]
    return [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:k]]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """This function concatenates the instructions, the retrieved chunks and the question.
    This is what we send to the model.
    """
    context = "\n\n".join(context_chunks)
    return (
        "Base your answer on the context below and nothing else. "
        "If the context does not contain the answer, say that you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def main() -> None:
    """Function that loads and embeds the docs once, then keeps answering your questions
    until you type q or an empty line.
    """
    chunks = load_chunks(Path(__file__).parent / "docs")
    vectors = embed(chunks)  # done only once, because embedding is the slow part and searching is fast
    print(f"Indexed {len(chunks)} chunks")
    while (question := input("\nQuestion: ").strip()) not in ("", "q"):  # an empty line or 'q' quits
        hits = retrieve(question, chunks, vectors, TOP_K)
        print(ask_llm(build_prompt(question, [chunks[i] for i, _ in hits])))


if __name__ == "__main__":
    try:
        main()
    except OllamaError as e:
        raise SystemExit(f"Error: {e}")  # print just the message, no traceback
